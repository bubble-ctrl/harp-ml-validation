"""
validators/merge/merge_validator.py — Merge API validation pipeline.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from models.slm2_merge_contract import infer_contract
from validators.merge.merge_pbt_runner import extract_merge_callable, run_pbt
from utils.ast_helpers import extract_merge_spec, structural_merge_check


class MergeVerdict(str, Enum):
    VALIDATED = "VALIDATED"
    REGRESSION = "REGRESSION"
    AMBIGUOUS = "AMBIGUOUS"
    NO_SMELL = "NO_SMELL"
    EXEC_ERROR = "EXEC_ERROR"


@dataclass
class MergeValidatorResult:
    verdict: MergeVerdict
    contract: Optional[dict] = None
    pbt_passed: Optional[bool] = None
    summary: str = ""
    confidence: str = ""


def validate_merge(original_code: str, refactored_code: str) -> MergeValidatorResult:
    # Structural check
    if structural_merge_check(original_code):
        return MergeValidatorResult(
            verdict=MergeVerdict.NO_SMELL,
            summary="All parameters already explicit.",
            confidence="HIGH"
        )

    # Infer contract
    contract = infer_contract(original_code, refactored_code)
    if contract is None:
        return MergeValidatorResult(
            verdict=MergeVerdict.AMBIGUOUS,
            summary="SLM‑2 failed to infer contract.",
            confidence="LOW"
        )

    # Structural comparison comes before execution. The desired contract was
    # inferred independently; the candidate is not allowed to redefine it.
    try:
        spec = extract_merge_spec(refactored_code)
    except (SyntaxError, ValueError) as exc:
        return MergeValidatorResult(
            verdict=MergeVerdict.EXEC_ERROR,
            contract=contract,
            summary=f"Candidate cannot be analyzed: {exc}",
            confidence="LOW",
        )

    mismatches = []
    if not spec.get("explicit_on"):
        mismatches.append("join keys are not explicit")
    if not spec.get("explicit_how"):
        mismatches.append("how is not explicit")
    if not spec.get("explicit_validate"):
        mismatches.append("validate is not explicit")
    for field in ("how", "left_on", "right_on", "validate"):
        expected = contract.get(field)
        actual = spec.get(field)
        if expected not in (None, "unknown") and actual != expected:
            mismatches.append(f"{field}: expected {expected!r}, got {actual!r}")
    if contract.get("suffixes_required") and "suffixes" not in spec:
        mismatches.append("explicit suffixes are required for overlapping columns")
    if mismatches:
        return MergeValidatorResult(
            verdict=MergeVerdict.REGRESSION,
            contract=contract,
            pbt_passed=False,
            summary="Structural contract mismatch: " + "; ".join(mismatches),
            confidence=contract.get("confidence", "medium"),
        )

    # PBT executes an isolated merge expression with injected DataFrames.
    try:
        func = extract_merge_callable(refactored_code)
        pbt_passed, msg = run_pbt(func, contract, original_code, runs=100)
    except Exception as e:
        return MergeValidatorResult(
            verdict=MergeVerdict.EXEC_ERROR,
            contract=contract,
            summary=f"Execution error: {e}",
            confidence="LOW"
        )

    if pbt_passed:
        verdict = MergeVerdict.VALIDATED
        summary = f"Contract validated. {msg}"
    else:
        verdict = MergeVerdict.REGRESSION
        summary = f"PBT failed. {msg}"

    return MergeValidatorResult(
        verdict=verdict,
        contract=contract,
        pbt_passed=pbt_passed,
        summary=summary,
        confidence=contract.get("confidence", "medium")
    )
