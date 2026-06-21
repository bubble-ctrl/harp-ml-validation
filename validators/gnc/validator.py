"""
validator.py — Streamlined GNC validator.

Pipeline now consists of:
  Stage 1 → AST intentionality check (FP / Ambiguous filter)
  Stage 4 → Differential behavioral equivalence (C1–C4)

Stages 2/2b/3 (gradient norms, hook check, stationarity) have been removed
because they effectively re-detect the smell, which is the responsibility
of the external detector (CodeSmile).

The validator answers only:
  "Did the refactoring break the training behavior?"

Usage:
    from validator import validate, FinalVerdict
    result = validate(before_snippet, after_snippet)
    print(result.verdict, result.summary)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from validators.gnc.stage1_intent import check_intentionality, Verdict as IntentVerdict
from validators.gnc.stage4_differential import compare_snippets, DiffVerdict


class FinalVerdict(str, Enum):
    VALIDATED    = "VALIDATED"    # Stage 4 passed → behavior preserved
    REGRESSION   = "REGRESSION"   # Stage 4 failed → behavior broken
    AMBIGUOUS    = "AMBIGUOUS"    # Stage 1: intentional accumulation
    NO_SMELL     = "NO_SMELL"     # Stage 1: zero_grad already present
    INCONCLUSIVE = "INCONCLUSIVE" # Stage 4 borderline
    EXEC_ERROR   = "EXEC_ERROR"   # Could not execute on scaffold


@dataclass
class ValidatorResult:
    verdict:    FinalVerdict = FinalVerdict.INCONCLUSIVE   # default
    summary:    str = ""                                   # default
    confidence: str = ""          # HIGH | MEDIUM | LOW

    # Stage 1
    intent_verdict:    Optional[str]   = None
    intent_reason:     Optional[str]   = None
    intent_confidence: Optional[float] = None

    # Stage 4 (differential)
    diff_verdict:      Optional[str]   = None
    diff_summary:      Optional[str]   = None
    c3_improvement:    Optional[float] = None
    c4_cosine:         Optional[float] = None

    before_losses: list[float] = field(default_factory=list)
    after_losses:  list[float] = field(default_factory=list)


def validate(
    before_snippet: str,
    after_snippet:  str,
    num_steps: int = 40,
    seed: int = 42,
) -> ValidatorResult:
    """
    Full validation pipeline for a GNC refactoring.

    Stage 1 filters false positives / intentional accumulation.
    Stage 4 checks behavioral equivalence (C1–C4).

    Returns VALIDATED iff Stage 4 says PRESERVED.
    """

    # ── Stage 1: intentionality gate ──────────────────────────────────────────
    intent = check_intentionality(before_snippet)

    if intent.verdict == IntentVerdict.AMBIGUOUS:
        return ValidatorResult(
            verdict=FinalVerdict.AMBIGUOUS,
            confidence="LOW",
            intent_verdict=intent.verdict.value,
            intent_reason=intent.reason,
            intent_confidence=intent.confidence,
            summary=(
                f"Stage 1: possible intentional accumulation "
                f"({intent.confidence:.0%} confidence) — {intent.reason}. "
                f"Human review required before accepting refactoring."
            )
        )

    if intent.verdict == IntentVerdict.NO_SMELL:
        return ValidatorResult(
            verdict=FinalVerdict.NO_SMELL,
            confidence="HIGH",
            intent_verdict=intent.verdict.value,
            intent_reason=intent.reason,
            intent_confidence=intent.confidence,
            summary=(
                f"Stage 1: zero_grad() already present in original code. "
                f"This instance may be a false positive. {intent.reason}"
            )
        )

    # ── Stage 4: differential behavioral check ────────────────────────────────
    diff = compare_snippets(before_snippet, after_snippet, num_steps=num_steps, seed=seed)

    result = ValidatorResult(
        intent_verdict=intent.verdict.value,
        intent_reason=intent.reason,
        intent_confidence=intent.confidence,
        diff_verdict=diff.verdict.value,
        diff_summary=diff.summary,
        before_losses=diff.before_losses,
        after_losses=diff.after_losses,
    )
    if diff.c3_improvement:
        result.c3_improvement = diff.c3_improvement.value
    if diff.c4_identity:
        result.c4_cosine = diff.c4_identity.value

    # ── Final verdict ──────────────────────────────────────────────────────────
    if diff.verdict == DiffVerdict.PRESERVED:
        result.verdict = FinalVerdict.VALIDATED
        result.confidence = "HIGH"
        result.summary = (
            f"Refactoring validated. Behavior preserved. "
            f"Loss improved {(result.c3_improvement or 0)*100:.1f}% in refactored code. "
            f"Step-0 gradient identity: cosine={result.c4_cosine:.4f}."
        )

    elif diff.verdict == DiffVerdict.REGRESSION:
        result.verdict = FinalVerdict.REGRESSION
        result.confidence = "HIGH"
        result.summary = (
            f"Regression detected — refactoring broke training behavior. "
            f"{diff.summary}"
        )

    else:  # INCONCLUSIVE
        result.verdict = FinalVerdict.INCONCLUSIVE
        result.confidence = "MEDIUM"
        result.summary = (
            f"Inconclusive — behavioral equivalence borderline. "
            f"{diff.summary}"
        )

    return result


def print_result(label: str, r: ValidatorResult):
    """Pretty-print a ValidatorResult."""
    icons = {
        "VALIDATED":  "✓",
        "REGRESSION": "✗",
        "AMBIGUOUS":  "?",
        "NO_SMELL":   "–",
        "INCONCLUSIVE": "~",
        "EXEC_ERROR": "!",
    }
    w = 70
    print("\n" + "─" * w)
    print(f"  {label}")
    print("─" * w)
    icon = icons.get(r.verdict.value, " ")
    print(f"  {icon} VERDICT    : {r.verdict.value}  [{r.confidence}]")
    print(f"    summary    : {r.summary}")

    print(f"\n  [Stage 1 — intentionality]")
    print(f"    {r.intent_verdict}  (conf {r.intent_confidence:.0%})  —  {r.intent_reason}")

    if r.diff_verdict:
        print(f"\n  [Stage 4 — behavior preserved?]")
        print(f"    diff    : {r.diff_verdict}")
        print(f"    C3 improvement : {(r.c3_improvement or 0)*100:.1f}%")
        print(f"    C4 cosine      : {r.c4_cosine:.6f}" if r.c4_cosine is not None else "    C4 cosine      : n/a")
        if r.before_losses and r.after_losses:
            print(f"    before  : {[round(l,3) for l in r.before_losses[:5]]}...")
            print(f"    after   : {[round(l,3) for l in r.after_losses[:5]]}...")


if __name__ == "__main__":
    SMELL = """
for epoch in range(num_epochs):
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
"""
    CORRECT_FIX = """
for epoch in range(num_epochs):
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
"""
    BAD_FIX = """
for epoch in range(num_epochs):
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()
"""
    INTENTIONAL = """
accumulation_steps = 4
for step in range(num_steps):
    out = model(X)
    loss = criterion(out, y) / accumulation_steps
    loss.backward()
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
"""
    ALREADY_FIXED = """
for epoch in range(num_epochs):
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
"""

    cases = {
        "1. correct fix":              (SMELL, CORRECT_FIX),
        "2. bad fix (zg after bw)":    (SMELL, BAD_FIX),
        "3. intentional accumulation": (INTENTIONAL, CORRECT_FIX),
        "4. already fixed (no smell)": (ALREADY_FIXED, CORRECT_FIX),
    }

    for label, (before, after) in cases.items():
        result = validate(before, after, num_steps=40)
        print_result(label, result)

    print("\n" + "─" * 70)
    print("  Done.")