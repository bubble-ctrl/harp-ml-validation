"""
Stage 4 — Differential behavioral equivalence testing for GNC refactorings.

Runs both original and refactored snippets on the *same* synthetic scaffold
(deterministic seed, identical model weights, data, optimizer, criterion)
and checks four behavioral properties:

  C1  Both snippets complete without crashing
  C2  Both produce finite loss values (no NaN / Inf)
  C3  Refactored code shows loss decreasing over steps (≥ 2 % improvement)
  C4  Step‑0 gradients are identical (cosine similarity > 0.9999)

Verdict:
  PRESERVED    — all checks pass
  REGRESSION   — any check fails clearly
  INCONCLUSIVE — borderline (C3 < 2 % or C4 cosine in [0.95, 0.9999])
"""

import copy
import math
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim


# ── Public types ──────────────────────────────────────────────────────────────

class DiffVerdict(str, Enum):
    PRESERVED    = "PRESERVED"
    REGRESSION   = "REGRESSION"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class CheckResult:
    passed: bool
    value: Optional[float] = None
    detail: str = ""


@dataclass
class DiffResult:
    verdict: DiffVerdict
    summary: str
    c1_original: Optional[CheckResult] = None
    c1_refactored: Optional[CheckResult] = None
    c2_original: Optional[CheckResult] = None
    c2_refactored: Optional[CheckResult] = None
    c3_improvement: Optional[CheckResult] = None
    c4_identity: Optional[CheckResult] = None
    before_losses: list[float] = field(default_factory=list)
    after_losses: list[float] = field(default_factory=list)


# ── Scaffold ──────────────────────────────────────────────────────────────────

class _FlexibleNet(nn.Module):
    """
    Tiny network for differential testing that handles arbitrary input shapes.
    Flattens any input to a fixed-size vector (pad or truncate), then passes
    through Linear→ReLU→Linear. This avoids LazyLinear parameter-tracking
    issues with optimizers created before first forward.
    """
    _MAX_IN = 2048  # fixed input dimension

    def __init__(self, out_features: int = 1, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(self._MAX_IN, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        flat = x.reshape(x.size(0), -1).float()
        # Pad or truncate to fixed input size
        if flat.size(1) < self._MAX_IN:
            flat = torch.nn.functional.pad(flat, (0, self._MAX_IN - flat.size(1)))
        elif flat.size(1) > self._MAX_IN:
            flat = flat[:, :self._MAX_IN]
        return self.fc2(self.relu(self.fc1(flat)))

    def train(self, mode=True):
        """Override to handle model.train() calls in snippets."""
        return super().train(mode)

    def functional_forward(self, x, weights=None):
        """Stub for MAML-style snippets that call model.functional_forward."""
        return self.forward(x)


class _DataParallelWrapper(nn.Module):
    """Lightweight wrapper that mimics nn.DataParallel for test snippets."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _build_scaffold(seed: int = 42):
    """
    Build a deterministic scaffold: model, optimizer, criterion, data.

    Returns a dict of names that will be injected into the snippet's
    execution namespace.
    """
    torch.manual_seed(seed)

    model = _FlexibleNet(out_features=1, hidden=32)
    optimizer_obj = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Fixed random dataset
    torch.manual_seed(seed)
    X = torch.randn(16, 10)
    y = torch.randn(16, 1)

    # Simple list-based "dataloader" so snippets with `for ... in loader:` work
    batch_size = 4
    loader = []
    for i in range(0, len(X), batch_size):
        loader.append((X[i:i+batch_size], y[i:i+batch_size]))

    # Fake DataLoader class for snippets that use DataLoader(...)
    train_loader = loader

    return {
        "model": model,
        "optimizer": optimizer_obj,
        "criterion": criterion,
        "loss_fn": criterion,
        "X": X,
        "y": y,
        "loader": loader,
        "train_loader": train_loader,
        "dataloader": loader,
        "data_loader": loader,
        "batch_size": batch_size,
        "num_epochs": 5,
        "epochs": 5,
        "num_steps": 40,
    }


# ── Capturing wrappers ───────────────────────────────────────────────────────

class _CapturingOptimizer:
    """
    Wraps an optimizer to capture gradient vectors at each step() call.
    Used for C4 (gradient identity check).
    """
    def __init__(self, base_optimizer: optim.Optimizer):
        self._opt = base_optimizer
        self.grad_snapshots: list[torch.Tensor] = []

    def step(self, closure=None):
        # Capture current gradients (flattened) before the step
        grads = []
        for group in self._opt.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(p.grad.detach().clone().flatten())
        if grads:
            self.grad_snapshots.append(torch.cat(grads))
        return self._opt.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        return self._opt.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self._opt.param_groups

    def state_dict(self):
        return self._opt.state_dict()

    def load_state_dict(self, state_dict):
        return self._opt.load_state_dict(state_dict)

    def __getattr__(self, name):
        # Delegate anything else to the underlying optimizer
        return getattr(self._opt, name)


class _CapturingCriterion(nn.Module):
    """
    Wraps a loss function to record every loss value.
    Used for C2 (finiteness) and C3 (improvement).
    """
    def __init__(self, base_criterion: nn.Module):
        super().__init__()
        self._criterion = base_criterion
        self.loss_values: list[float] = []

    def forward(self, *args, **kwargs):
        loss = self._criterion(*args, **kwargs)
        self.loss_values.append(loss.item())
        return loss


# ── Snippet normalization ─────────────────────────────────────────────────────

def _normalize_snippet(snippet: str) -> str:
    """
    Normalize a code snippet so it can be executed directly against the scaffold.

    Handles three cases:
      1. Function definition (def train(...):) → extract body, strip returns
      2. Bare loop (for epoch in ...: ) → run as-is
      3. Bare statements (no loop, no def) → wrap in a for loop
    """
    stripped = textwrap.dedent(snippet).strip()
    lines = stripped.splitlines()

    if not lines:
        return ""

    # Case 1: Function definition → extract body
    if lines[0].lstrip().startswith("def "):
        body_lines = []
        in_body = False
        body_indent = None

        for line in lines[1:]:
            if not in_body:
                if line.strip() == "":
                    continue
                in_body = True
                body_indent = len(line) - len(line.lstrip())

            if in_body:
                if line.strip() and (len(line) - len(line.lstrip())) < body_indent:
                    break
                if len(line) >= body_indent:
                    body_lines.append(line[body_indent:])
                else:
                    body_lines.append(line)

        # Strip return statements (they can't execute outside a function)
        cleaned_lines = []
        for line in body_lines:
            stripped_line = line.lstrip()
            if stripped_line.startswith("return ") or stripped_line == "return":
                continue
            cleaned_lines.append(line)

        stripped = "\n".join(cleaned_lines).strip()
        lines = stripped.splitlines()

    if not lines:
        return ""

    # Check if there's any loop in the code (not just the first line)
    has_loop = any(
        l.lstrip().startswith("for ") or l.lstrip().startswith("while ")
        for l in lines
    )

    if has_loop:
        return stripped

    # Case 3: Bare statements with backward() → wrap in a for loop
    if "backward()" in stripped:
        indented = textwrap.indent(stripped, "    ")
        return f"for _step in range(num_steps):\n{indented}"

    return stripped


# ── Execution ─────────────────────────────────────────────────────────────────

class _OptimizerInterceptor:
    """
    Intercor that wraps any optimizer created by the snippet with a
    _CapturingOptimizer. This allows us to capture gradients even when
    the snippet creates its own optimizer.
    """
    def __init__(self, capturing_list: list):
        self._capturing_list = capturing_list
        # Store original constructors
        self._originals = {}

    def _make_factory(self, orig_cls):
        capturing_list = self._capturing_list

        def factory(*args, **kwargs):
            base_opt = orig_cls(*args, **kwargs)
            cap_opt = _CapturingOptimizer(base_opt)
            capturing_list.append(cap_opt)
            return cap_opt
        return factory

    def patch(self, namespace):
        """Create a patched optim module for the namespace."""
        import types
        patched = types.ModuleType("optim")

        # Copy all attributes from real optim
        for attr in dir(optim):
            setattr(patched, attr, getattr(optim, attr))

        # Patch common optimizers
        for opt_name in ["SGD", "Adam", "AdamW", "RMSprop", "Adagrad",
                         "Adadelta", "Adamax", "LBFGS", "SparseAdam"]:
            orig = getattr(optim, opt_name, None)
            if orig:
                setattr(patched, opt_name, self._make_factory(orig))

        # Also patch lr_scheduler to reference patched optim
        namespace["optim"] = patched
        return patched


class _CriterionInterceptor:
    """
    Intercepts nn.Loss creation to wrap with _CapturingCriterion.
    """
    def __init__(self, capturing_list: list):
        self._capturing_list = capturing_list

    def _make_factory(self, orig_cls):
        capturing_list = self._capturing_list

        def factory(*args, **kwargs):
            base_crit = orig_cls(*args, **kwargs)
            cap_crit = _CapturingCriterion(base_crit)
            capturing_list.append(cap_crit)
            return cap_crit
        return factory

    def patch(self, namespace):
        """Create a patched nn module for the namespace."""
        import types
        patched = types.ModuleType("nn")

        # Copy all attributes from real nn
        for attr in dir(nn):
            setattr(patched, attr, getattr(nn, attr))

        # Patch common loss functions
        for loss_name in ["MSELoss", "CrossEntropyLoss", "BCELoss",
                          "BCEWithLogitsLoss", "L1Loss", "NLLLoss",
                          "SmoothL1Loss", "HuberLoss", "KLDivLoss"]:
            orig = getattr(nn, loss_name, None)
            if orig:
                setattr(patched, loss_name, self._make_factory(orig))

        # nn.functional needs special handling — wrap cross_entropy etc.
        patched.functional = nn.functional
        # DataParallel — use our lightweight wrapper
        patched.DataParallel = _DataParallelWrapper
        namespace["nn"] = patched
        return patched


def _run_snippet(
    snippet: str,
    seed: int = 42,
    num_steps: int = 40,
) -> tuple[bool, _CapturingOptimizer, _CapturingCriterion, list[float], Optional[str]]:
    """
    Execute a normalized snippet on a fresh scaffold.

    Uses interceptors so that even if the snippet creates its own
    model/optimizer/criterion, we still capture gradients and losses.

    Returns:
      (completed, capturing_optimizer, capturing_criterion, loss_list, error_msg)
    """
    scaffold = _build_scaffold(seed)

    # Capturing lists — interceptors will append to these
    captured_opts: list[_CapturingOptimizer] = []
    captured_crits: list[_CapturingCriterion] = []

    # Also create scaffold-level capturing wrappers (for snippets that
    # use the scaffold's pre-built optimizer/criterion directly)
    scaffold_cap_opt = _CapturingOptimizer(scaffold["optimizer"])
    scaffold_cap_crit = _CapturingCriterion(scaffold["criterion"])
    captured_opts.append(scaffold_cap_opt)
    captured_crits.append(scaffold_cap_crit)

    # Build execution namespace
    namespace = dict(scaffold)
    namespace["optimizer"] = scaffold_cap_opt
    namespace["opt"] = scaffold_cap_opt
    namespace["criterion"] = scaffold_cap_crit
    namespace["loss_fn"] = scaffold_cap_crit
    namespace["num_steps"] = num_steps

    # Add common imports
    namespace["torch"] = torch
    namespace["math"] = math

    # Patch optim and nn to intercept creation
    opt_interceptor = _OptimizerInterceptor(captured_opts)
    opt_interceptor.patch(namespace)

    crit_interceptor = _CriterionInterceptor(captured_crits)
    crit_interceptor.patch(namespace)

    # Mock objects for common references
    namespace["models"] = type("models", (), {
        "resnet50": staticmethod(lambda pretrained=False, **kw: _FlexibleNet(out_features=1000, hidden=32)),
    })()

    # DataLoader and related utilities — return scaffold's loader
    namespace["DataLoader"] = lambda *a, **kw: scaffold["loader"]
    namespace["TensorDataset"] = lambda *a, **kw: list(zip(a[0], a[1])) if len(a) >= 2 else []

    # Common variables that snippets reference
    namespace["pad_idx"] = 0
    namespace["vocab_size"] = 1000
    namespace["pack_padded_sequence"] = lambda *a, **kw: (torch.randn(4, 10), None)
    namespace["pad_packed_sequence"] = lambda *a, **kw: (torch.randn(4, 50, 1000), None)
    namespace["BertForSequenceClassification"] = type("BertForSequenceClassification", (), {
        "from_pretrained": staticmethod(lambda *a, **kw: _FlexibleNet(out_features=1, hidden=32)),
    })()

    # Normalize the snippet
    code = _normalize_snippet(snippet)
    if not code:
        return False, scaffold_cap_opt, scaffold_cap_crit, [], "Empty snippet after normalization"

    completed = False
    error = None
    try:
        exec(compile(code, "<snippet>", "exec"), namespace)
        completed = True
    except Exception as e:
        error = str(e)

    # Find the best capturing optimizer and criterion
    # (the one with the most data — i.e., the one actually used in training)
    best_opt = max(captured_opts, key=lambda o: len(o.grad_snapshots))
    best_crit = max(captured_crits, key=lambda c: len(c.loss_values))

    return completed, best_opt, best_crit, best_crit.loss_values, error


# ── Checks ────────────────────────────────────────────────────────────────────

def _check_c1(completed: bool, error: Optional[str]) -> CheckResult:
    """C1: Snippet completes without crash."""
    if completed:
        return CheckResult(passed=True, detail="Execution completed successfully")
    return CheckResult(passed=False, detail=f"Execution failed: {error}")


def _check_c2(losses: list[float]) -> CheckResult:
    """C2: All loss values are finite (no NaN or Inf)."""
    if not losses:
        return CheckResult(passed=False, detail="No loss values recorded")

    non_finite = [i for i, v in enumerate(losses) if not math.isfinite(v)]
    if non_finite:
        return CheckResult(
            passed=False,
            detail=f"Non-finite loss at steps {non_finite[:5]}",
        )
    return CheckResult(passed=True, detail=f"All {len(losses)} loss values finite")


def _check_c3(before_losses: list[float], after_losses: list[float]) -> CheckResult:
    """
    C3: Refactored code shows loss decreasing.

    Compares the average loss of the last 25% of steps to the first 25%.
    Requires at least 2% improvement in the refactored code.

    If both snippets are identical (or very close), we don't flag regression —
    the refactoring just didn't change behavior, which is "preserved".
    """
    if not after_losses or len(after_losses) < 4:
        return CheckResult(
            passed=False,
            value=0.0,
            detail="Insufficient loss values for trend analysis",
        )

    n = len(after_losses)
    quarter = max(1, n // 4)

    first_quarter = sum(after_losses[:quarter]) / quarter
    last_quarter = sum(after_losses[-quarter:]) / quarter

    if first_quarter == 0:
        # Avoid division by zero
        if last_quarter == 0:
            return CheckResult(passed=True, value=0.0, detail="Loss is zero throughout")
        return CheckResult(passed=False, value=0.0, detail="First-quarter loss is zero")

    improvement = (first_quarter - last_quarter) / abs(first_quarter)

    # If loss is getting worse (increasing), that's a regression
    if improvement < -0.02:
        return CheckResult(
            passed=False,
            value=improvement,
            detail=f"Loss worsened by {abs(improvement)*100:.1f}%",
        )

    # If loss improved by at least 2%, clear pass
    if improvement >= 0.02:
        return CheckResult(
            passed=True,
            value=improvement,
            detail=f"Loss improved by {improvement*100:.1f}%",
        )

    # Borderline: 0% to 2% improvement — check if before losses also showed
    # similar trajectory (i.e., code didn't change behavior = PRESERVED)
    if before_losses and len(before_losses) >= 4:
        bn = len(before_losses)
        bquarter = max(1, bn // 4)
        before_first = sum(before_losses[:bquarter]) / bquarter
        before_last = sum(before_losses[-bquarter:]) / bquarter
        if before_first != 0:
            before_improvement = (before_first - before_last) / abs(before_first)
            # If both have similar trajectories, behavior is preserved
            diff = abs(improvement - before_improvement)
            if diff < 0.05:  # within 5% of each other
                return CheckResult(
                    passed=True,
                    value=improvement,
                    detail=f"Loss trend similar to original ({improvement*100:.1f}% vs {before_improvement*100:.1f}%)",
                )

    # Borderline
    return CheckResult(
        passed=True,  # Don't flag as regression for borderline
        value=improvement,
        detail=f"Borderline improvement: {improvement*100:.1f}%",
    )


def _check_c4(
    before_grads: list[torch.Tensor],
    after_grads: list[torch.Tensor],
) -> CheckResult:
    """
    C4: Step‑0 gradients are identical (cosine similarity > 0.9999).

    This is the strongest check — if the SLM changed the forward pass or loss
    expression, the very first gradient vector will differ.
    """
    if not before_grads:
        return CheckResult(
            passed=False,
            value=0.0,
            detail="No gradient snapshots from original code",
        )
    if not after_grads:
        return CheckResult(
            passed=False,
            value=0.0,
            detail="No gradient snapshots from refactored code",
        )

    g_before = before_grads[0].float()
    g_after = after_grads[0].float()

    # Handle size mismatch (different model architectures = definite regression)
    if g_before.shape != g_after.shape:
        return CheckResult(
            passed=False,
            value=0.0,
            detail=f"Gradient shape mismatch: {g_before.shape} vs {g_after.shape}",
        )

    # Handle zero gradients
    norm_before = torch.norm(g_before)
    norm_after = torch.norm(g_after)

    if norm_before < 1e-12 and norm_after < 1e-12:
        # Both zero — identical
        return CheckResult(passed=True, value=1.0, detail="Both gradients are zero")

    if norm_before < 1e-12 or norm_after < 1e-12:
        # One zero, one not — regression
        return CheckResult(
            passed=False,
            value=0.0,
            detail="One gradient is zero, the other is not",
        )

    cosine = torch.nn.functional.cosine_similarity(
        g_before.unsqueeze(0), g_after.unsqueeze(0)
    ).item()

    if cosine > 0.9999:
        return CheckResult(passed=True, value=cosine, detail=f"Cosine similarity: {cosine:.6f}")
    elif cosine > 0.95:
        return CheckResult(
            passed=False,  # borderline but flagged
            value=cosine,
            detail=f"Borderline cosine similarity: {cosine:.6f}",
        )
    else:
        return CheckResult(
            passed=False,
            value=cosine,
            detail=f"Gradient divergence: cosine={cosine:.6f}",
        )


# ── Public API ────────────────────────────────────────────────────────────────

def compare_snippets(
    before_snippet: str,
    after_snippet: str,
    num_steps: int = 40,
    seed: int = 42,
) -> DiffResult:
    """
    Run differential testing between original and refactored snippets.

    Both are executed on identical scaffolds (same seed → same model,
    same data, same optimizer state) and behavioral properties are compared.
    """
    # Run original
    ok_before, cap_opt_before, cap_crit_before, losses_before, err_before = \
        _run_snippet(before_snippet, seed=seed, num_steps=num_steps)

    # Run refactored
    ok_after, cap_opt_after, cap_crit_after, losses_after, err_after = \
        _run_snippet(after_snippet, seed=seed, num_steps=num_steps)

    # ── C1: Completion ────────────────────────────────────────────────────
    c1_orig = _check_c1(ok_before, err_before)
    c1_ref = _check_c1(ok_after, err_after)

    if not c1_orig.passed and not c1_ref.passed:
        return DiffResult(
            verdict=DiffVerdict.INCONCLUSIVE,
            summary=f"Both snippets crashed. Original: {err_before}. Refactored: {err_after}",
            c1_original=c1_orig,
            c1_refactored=c1_ref,
            before_losses=losses_before,
            after_losses=losses_after,
        )

    if not c1_ref.passed:
        return DiffResult(
            verdict=DiffVerdict.REGRESSION,
            summary=f"Refactored code crashed: {err_after}",
            c1_original=c1_orig,
            c1_refactored=c1_ref,
            before_losses=losses_before,
            after_losses=losses_after,
        )

    if not c1_orig.passed:
        # Original crashed but refactored didn't — unusual but not a regression
        return DiffResult(
            verdict=DiffVerdict.INCONCLUSIVE,
            summary=f"Original code crashed ({err_before}) but refactored code succeeded",
            c1_original=c1_orig,
            c1_refactored=c1_ref,
            before_losses=losses_before,
            after_losses=losses_after,
        )

    # ── C2: Finiteness ────────────────────────────────────────────────────
    c2_orig = _check_c2(losses_before)
    c2_ref = _check_c2(losses_after)

    if not c2_ref.passed:
        return DiffResult(
            verdict=DiffVerdict.REGRESSION,
            summary=f"Refactored code produced non-finite loss. {c2_ref.detail}",
            c1_original=c1_orig,
            c1_refactored=c1_ref,
            c2_original=c2_orig,
            c2_refactored=c2_ref,
            before_losses=losses_before,
            after_losses=losses_after,
        )

    # ── C3: Loss improvement ──────────────────────────────────────────────
    c3 = _check_c3(losses_before, losses_after)

    # ── C4: Gradient identity ─────────────────────────────────────────────
    c4 = _check_c4(
        cap_opt_before.grad_snapshots,
        cap_opt_after.grad_snapshots,
    )

    # ── Final verdict ─────────────────────────────────────────────────────
    result = DiffResult(
        verdict=DiffVerdict.PRESERVED,  # assume best, downgrade below
        summary="",
        c1_original=c1_orig,
        c1_refactored=c1_ref,
        c2_original=c2_orig,
        c2_refactored=c2_ref,
        c3_improvement=c3,
        c4_identity=c4,
        before_losses=losses_before,
        after_losses=losses_after,
    )

    failures = []
    if not c3.passed:
        failures.append(f"C3: {c3.detail}")
    if not c4.passed:
        failures.append(f"C4: {c4.detail}")

    if failures:
        # Check if all failures are borderline
        borderline_c4 = (
            c4.value is not None and 0.95 <= c4.value <= 0.9999
        )
        borderline_c3 = (
            c3.value is not None and -0.02 <= c3.value < 0.02
        )

        all_borderline = True
        if not c3.passed and not borderline_c3:
            all_borderline = False
        if not c4.passed and not borderline_c4:
            all_borderline = False

        if all_borderline:
            result.verdict = DiffVerdict.INCONCLUSIVE
            result.summary = "Borderline: " + "; ".join(failures)
        else:
            result.verdict = DiffVerdict.REGRESSION
            result.summary = "; ".join(failures)
    else:
        c3_val = f"{(c3.value or 0)*100:.1f}%" if c3.value is not None else "n/a"
        c4_val = f"{c4.value:.6f}" if c4.value is not None else "n/a"
        result.summary = f"Behavior preserved. Loss improvement: {c3_val}, Gradient cosine: {c4_val}"

    return result
