"""
Stage 1 — AST intentionality checker for GNC smell.

Returns one of three verdicts:
  GENUINE   — no signs of intentional accumulation, proceed to Stage 2
  AMBIGUOUS — structural signals suggest accumulation may be intentional
  NO_SMELL  — zero_grad() is present and correctly placed, nothing to validate
"""

import ast
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Verdict(str, Enum):
    GENUINE   = "GENUINE"
    AMBIGUOUS = "AMBIGUOUS"
    NO_SMELL  = "NO_SMELL"


@dataclass
class IntentResult:
    verdict: Verdict
    reason: str
    confidence: float   # 0.0 - 1.0, how sure we are about the verdict


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_all_calls(tree: ast.AST, method: str) -> list[ast.Call]:
    """Return all Call nodes whose func is an attribute named `method`."""
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
    ]


def _get_optimizer_names(tree: ast.AST) -> set[str]:
    """
    Heuristic: find variable names that are likely optimizers.
    Looks for assignments like: optimizer = optim.Adam(...)
    or any name containing 'optim' / 'optimizer'.
    """
    names = set()
    optim_constructors = {
        "Adam", "SGD", "AdamW", "RMSprop", "Adagrad",
        "Adadelta", "LBFGS", "SparseAdam", "Adamax",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # direct name heuristic
                    if "optim" in target.id.lower():
                        names.add(target.id)
                    # rhs is optim.Adam(...) or torch.optim.Adam(...)
                    if isinstance(node.value, ast.Call):
                        func = node.value.func
                        if isinstance(func, ast.Attribute) and func.attr in optim_constructors:
                            names.add(target.id)
    return names


def _nesting_depth(node: ast.AST, target_lineno: int, tree: ast.AST) -> int:
    """
    Count how many For/While loops enclose the line at target_lineno.
    Simple linear walk — good enough for training loop patterns.
    """
    depth = 0
    for n in ast.walk(tree):
        if isinstance(n, (ast.For, ast.While)):
            # check if target line is inside this loop's body range
            if hasattr(n, 'lineno') and hasattr(n, 'end_lineno'):
                if n.lineno <= target_lineno <= (n.end_lineno or target_lineno):
                    depth += 1
    return depth


def _has_accumulation_step_pattern(tree: ast.AST) -> tuple[bool, str]:
    """
    Detect the canonical intentional-accumulation gating pattern:
      if (step + 1) % N == 0:
          optimizer.step()
          optimizer.zero_grad()

    IMPORTANT: we require the if-body to contain an actual optimizer call
    (zero_grad or optimizer.step), NOT just any statement with 'step' in it.
    This prevents triggering on logging conditionals like:
      if step % report_frequency == 0:
          print(...)
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            cond_src = ast.dump(node.test)
            if "Mod" not in cond_src and "%" not in ast.unparse(node.test):
                continue

            # Walk the if-body and check for actual optimizer method calls
            has_optimizer_call = False
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    method = child.func.attr
                    # must be zero_grad, OR a .step() call on something that
                    # looks like an optimizer (not a scheduler or lr_ object)
                    if method == "zero_grad":
                        has_optimizer_call = True
                        break
                    if method == "step":
                        caller = ""
                        if isinstance(child.func.value, ast.Name):
                            caller = child.func.value.id.lower()
                        # exclude known non-optimizer step callers
                        if not any(s in caller for s in ("sched", "lr_", "scaler")):
                            has_optimizer_call = True
                            break

            if has_optimizer_call:
                return True, f"conditional optimizer.step/zero_grad with modulo at line {node.lineno}"
    return False, ""


def _has_accumulation_variable(tree: ast.AST) -> tuple[bool, str]:
    """Look for variables named like accumulation_steps, grad_accum_steps etc."""
    accum_keywords = {
        "accumulation_steps", "accum_steps", "grad_accum",
        "gradient_accumulation_steps", "accumulate_grad_batches",
        "update_freq", "accum_iter",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in accum_keywords:
            return True, f"accumulation variable '{node.id}' found at line {node.lineno}"
        if isinstance(node, ast.arg) and node.arg in accum_keywords:
            return True, f"accumulation parameter '{node.arg}' in function signature"
    return False, ""


def _has_nested_backward(tree: ast.AST) -> tuple[bool, str]:
    """
    Detect backward() calls that are nested deeper than optimizer.step().

    The key distinction:
      Standard epoch+batch loop: backward() and optimizer.step() are at the
      SAME nesting depth (both inside the batch loop). This is normal training
      structure — NOT intentional accumulation.

      Intentional accumulation: backward() is deeper than optimizer.step()
      (step() is in the outer loop, backward() is in the inner minibatch loop).
      This means gradients accumulate across inner iterations before a single
      outer update — genuinely intentional.
    """
    backward_calls = _get_all_calls(tree, "backward")
    step_calls     = _get_all_calls(tree, "step")

    for bw_call in backward_calls:
        bw_depth = _nesting_depth(bw_call, bw_call.lineno, tree)
        if bw_depth < 2:
            continue  # not nested at all — not this pattern

        # Find the shallowest optimizer.step() call — exclude lr_scheduler.step()
        # and similar non-optimizer calls which appear at shallower depth
        # and would cause false positives (standard epoch+batch loops).
        optimizer_step_calls = []
        for st in step_calls:
            caller_name = ""
            if isinstance(st.func.value, ast.Name):
                caller_name = st.func.value.id.lower()
            elif isinstance(st.func.value, ast.Attribute):
                caller_name = st.func.value.attr.lower()
            if any(s in caller_name for s in ("sched", "scheduler", "lr_")):
                continue
            optimizer_step_calls.append(st)

        step_depths = [
            _nesting_depth(st, st.lineno, tree)
            for st in optimizer_step_calls
        ]

        if not step_depths:
            # No optimizer.step() at all — unusual, treat as ambiguous
            return True, f"backward() at line {bw_call.lineno} is {bw_depth} loops deep with no optimizer.step() found"

        min_step_depth = min(step_depths)

        if bw_depth > min_step_depth:
            # backward is DEEPER than step — intentional accumulation pattern
            return True, (
                f"backward() at line {bw_call.lineno} is {bw_depth} loops deep "
                f"but optimizer.step() is only {min_step_depth} loops deep — "
                f"accumulation across inner loop iterations is intentional"
            )
        # bw_depth == min_step_depth: standard epoch+batch loop, not intentional

    return False, ""


def _has_multiple_optimizers(tree: ast.AST) -> tuple[bool, str]:
    """
    Detect multiple optimizer instances with separate zero_grad calls (Case 3).
    """
    optimizer_names = _get_optimizer_names(tree)
    zero_grad_callers = set()
    for call in _get_all_calls(tree, "zero_grad"):
        if isinstance(call.func.value, ast.Name):
            zero_grad_callers.add(call.func.value.id)

    # if zero_grad is called on more than one distinct name
    if len(zero_grad_callers) > 1:
        return True, f"zero_grad called on multiple objects: {zero_grad_callers}"

    # if we found multiple optimizer-like names but only one zero_grad caller
    if len(optimizer_names) > 1 and len(zero_grad_callers) == 1:
        missing = optimizer_names - zero_grad_callers
        return True, f"multiple optimizers {optimizer_names}, zero_grad only on {zero_grad_callers}"

    return False, ""


def _has_no_optimizer(tree: ast.AST) -> tuple[bool, str]:
    """
    Detect functions where backward() is used with no optimizer present.

    If there is no optimizer anywhere in the code, backward() is being used
    for a non-training purpose — adversarial perturbations, profiling, gradient
    analysis, custom loss gradient chains, etc. The GNC smell definition does
    not apply: there are no parameter updates to corrupt.

    Examples from real experimental kit:
      - deepfool.py: backward() computes input gradients for adversarial attack
      - torch_utils.py profile(): backward() times the backward pass
      - CachedMultipleNegativesRankingLoss._backward_hook: custom gradient chain
    """
    # Check for any optimizer-like variable or zero_grad call
    optimizer_names = _get_optimizer_names(tree)
    zero_grad_calls = _get_all_calls(tree, "zero_grad")

    # Also check for optimizer.step() calls (excluding schedulers)
    step_calls = _get_all_calls(tree, "step")
    optimizer_step_calls = []
    for st in step_calls:
        caller = ""
        if isinstance(st.func.value, ast.Name):
            caller = st.func.value.id.lower()
        elif isinstance(st.func.value, ast.Attribute):
            caller = st.func.value.attr.lower()
        if not any(s in caller for s in ("sched", "lr_", "scaler")):
            optimizer_step_calls.append(st)

    if not optimizer_names and not zero_grad_calls and not optimizer_step_calls:
        # No optimizer signals at all — backward() is not for training
        backward_calls = _get_all_calls(tree, "backward")
        if backward_calls:
            return True, (
                "no optimizer present — backward() is used for a non-training "
                "purpose (adversarial perturbation, profiling, custom gradient chain)"
            )
    return False, ""
    """Check if zero_grad() exists at all in the snippet."""
    calls = _get_all_calls(tree, "zero_grad")
    if calls:
        lines = [c.lineno for c in calls]
        return True, f"zero_grad() found at lines {lines}"
    return False, ""


def _zero_grad_present(tree: ast.AST) -> tuple[bool, str]:
    """Check if zero_grad() exists at all in the snippet."""
    calls = _get_all_calls(tree, "zero_grad")
    if calls:
        lines = [c.lineno for c in calls]
        return True, f"zero_grad() found at lines {lines}"
    return False, ""


# ── main entry ────────────────────────────────────────────────────────────────

def check_intentionality(source: str) -> IntentResult:
    """
    Parse source and return an IntentResult.

    Checks (in order, short-circuit on AMBIGUOUS):
      1. zero_grad present → NO_SMELL (nothing to validate)
         EXCEPT: zero_grad present but only in a conditional (intentional pattern)
      2. accumulation variable names
      3. conditional modulo gating
      4. nested backward (>= 2 loops deep)
      5. multiple optimizers with asymmetric zero_grad
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return IntentResult(Verdict.AMBIGUOUS, f"syntax error: {e}", 0.0)

    # Check zero_grad presence
    zg_present, zg_reason = _zero_grad_present(tree)

    # Check all intentionality signals
    signals = []

    # Most definitive check first: if no optimizer exists, backward() is not
    # being used for training at all — smell definition does not apply.
    has_no_opt, no_opt_reason = _has_no_optimizer(tree)
    if has_no_opt:
        return IntentResult(
            Verdict.AMBIGUOUS,
            f"No optimizer present — {no_opt_reason}",
            0.95,
        )

    has_accum_var, accum_var_reason = _has_accumulation_variable(tree)
    if has_accum_var:
        signals.append(accum_var_reason)

    has_accum_pattern, accum_pattern_reason = _has_accumulation_step_pattern(tree)
    if has_accum_pattern:
        signals.append(accum_pattern_reason)

    has_nested_bw, nested_bw_reason = _has_nested_backward(tree)
    if has_nested_bw:
        signals.append(nested_bw_reason)

    has_multi_opt, multi_opt_reason = _has_multiple_optimizers(tree)
    if has_multi_opt:
        signals.append(multi_opt_reason)

    # Decision logic
    if signals:
        # Intentionality signals found — ambiguous regardless of zero_grad
        reason = "Possible intentional accumulation: " + "; ".join(signals)
        confidence = min(0.5 + 0.15 * len(signals), 0.9)
        return IntentResult(Verdict.AMBIGUOUS, reason, confidence)

    if zg_present:
        # zero_grad present, no intentionality signals → likely correct or false positive
        return IntentResult(Verdict.NO_SMELL, f"zero_grad present ({zg_reason})", 0.85)

    # No zero_grad, no intentionality signals → genuine smell
    return IntentResult(
        Verdict.GENUINE,
        "zero_grad() absent with no signs of intentional accumulation",
        0.9,
    )


# ── smoke tests ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = {
        "genuine_smell": """
for epoch in range(num_epochs):
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
""",
        "false_positive_fig7": """
for epoch in range(epochs):
    optimizer.zero_grad()
    out = net(torch_X)
    loss = criterion(out, torch_y)
    loss.backward()
    optimizer.step()
""",
        "intentional_accum_variable": """
accumulation_steps = 4
for step, (X, y) in enumerate(loader):
    loss = criterion(model(X), y) / accumulation_steps
    loss.backward()
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
""",
        "nested_hook": """
def backward_hook(grad_output, sentence_features, loss_obj):
    for sentence_feature, grad in zip(sentence_features, loss_obj.cache):
        for (reps_mb, _), grad_mb in zip(loss_obj.embed_minibatch_iter(), grad):
            surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
            surrogate.backward()
""",
        "multi_optimizer_gan": """
optimizer_G.zero_grad()
loss_G.backward()
optimizer_G.step()

loss_D.backward()
optimizer_D.step()
optimizer_D.zero_grad()
""",
        "gp_models_genuine_smell": """
def main(args):
    adam = torch.optim.Adam(gp.parameters(), lr=args.init_learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=gamma)
    for step in range(args.num_steps):
        loss = -gp.log_prob(data).sum() / T_train
        loss.backward()
        adam.step()
        scheduler.step()
        if step % report_frequency == 0 or step == args.num_steps - 1:
            print("[step %03d]  loss: %.3f" % (step, loss.item()))
""",
        "deepfool_no_optimizer": """
def deepfool(model, image, num_classes, overshoot, max_iter):
    x = copy.deepcopy(image).requires_grad_(True)
    for i in range(max_iter):
        fs[0, output[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()
        for k in range(1, num_classes):
            zero_gradients(x)
            fs[0, output[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()
""",
        "torch_utils_profiler": """
def profile(input, ops, n=10):
    for x in input:
        for m in ops:
            for _ in range(n):
                y = m(x)
                _ = y.sum().backward()
""",
    }

    for name, src in cases.items():
        result = check_intentionality(src)
        print(f"\n[{name}]")
        print(f"  verdict    : {result.verdict.value}")
        print(f"  confidence : {result.confidence:.2f}")
        print(f"  reason     : {result.reason}")