"""
validators/merge/merge_validator.py — Merge API validation pipeline.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from models.slm2_merge_contract import infer_contract
from validators.merge.merge_pbt_runner import run_pbt
from utils.ast_helpers import structural_merge_check, extract_function


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

    # Run PBT
    try:
        func = extract_function(refactored_code)
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