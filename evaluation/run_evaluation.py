"""
evaluation/run_evaluation.py — Unified evaluation runner for GNC + Merge.
"""

import sys
import json
from dataclasses import dataclass
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, ".")

# Import validators
from validators.gnc.validator import validate as validate_gnc, FinalVerdict as GNCVerdict
from validators.merge.merge_validator import validate_merge, MergeVerdict

# Import dataset
from data.manual_test_dataset import TEST_CASES
from utils.ast_helpers import structural_merge_check


@dataclass
class EvalResult:
    case_id: str
    smell_type: str
    expected_verdict: str
    actual_verdict: str
    passed: bool
    summary: str


def run_evaluation(use_slm_generation: bool = False, limit: int = None) -> List[EvalResult]:
    """
    Run all test cases.
    If use_slm_generation is True, SLM‑1 generates the refactoring from scratch.
    Otherwise, the pre‑defined after_correct / after_wrong are used.
    """
    results = []

    cases = list(TEST_CASES.items())[:limit] if limit is not None else TEST_CASES.items()
    for case_id, case in cases:
        smell_type = case.get("smell_type")
        before = case["before"]
        expected = case["expected_validator"]
        # REGRESSION labels describe supplied after_wrong fixtures. During
        # generation, SLM-1 is being asked to produce a correct repair instead.
        if use_slm_generation and expected == "REGRESSION":
            expected = "VALIDATED"

        # Determine which refactored code to use
        if use_slm_generation:
            # Call SLM‑1 to generate refactoring
            if smell_type == "merge" and structural_merge_check(before):
                refactored = before
            elif smell_type == "gnc":
                from models.slm_refactor import refactor_gnc
                refactor_result = refactor_gnc(before)
                if not refactor_result.success:
                    results.append(EvalResult(case_id, smell_type, expected,
                                              "GENERATION_ERROR", False,
                                              refactor_result.error or "SLM-1 generation failed"))
                    continue
                refactored = refactor_result.refactored_code
                # SLM‑1 might return false positive
                if refactor_result.is_false_positive:
                    actual = "NO_SMELL" if smell_type == "gnc" else "NO_SMELL"
                    passed = actual == expected
                    results.append(EvalResult(case_id, smell_type, expected, actual, passed, "SLM‑1 flagged FP"))
                    continue
            else:  # genuine merge smell
                from models.slm_refactor import refactor_merge
                refactor_result = refactor_merge(before)
                if not refactor_result.success:
                    results.append(EvalResult(case_id, smell_type, expected,
                                              "GENERATION_ERROR", False,
                                              refactor_result.error or "SLM-1 generation failed"))
                    continue
                refactored = refactor_result.refactored_code
        else:
            # Use the pre-defined refactoring based on expected verdict
            if expected in ("REGRESSION",):
                # Regression test cases: use the intentionally wrong refactoring
                refactored = case.get("after_wrong", "")
            elif expected in ("VALIDATED",):
                # Correct refactoring cases
                refactored = case.get("after_correct", "")
            else:
                refactored = ""

            if not refactored:
                # FP / AMBIGUOUS / NO_SMELL cases: no refactoring needed
                # Validator should short-circuit on the original code
                refactored = before

        # Run the appropriate validator
        if smell_type == "gnc":
            from validators.gnc.validator import validate as gnc_validate
            val_result = gnc_validate(before, refactored, num_steps=40)
            actual = val_result.verdict.value
            summary = val_result.summary
        else:  # merge
            val_result = validate_merge(before, refactored)
            actual = val_result.verdict.value
            summary = val_result.summary

        passed = (actual == expected)
        results.append(EvalResult(case_id, smell_type, expected, actual, passed, summary))

    return results


def print_summary(results: List[EvalResult]):
    print("\n" + "=" * 110)
    print(f"{'Case ID':<20} {'Type':<8} {'Expected':<16} {'Actual':<16} {'Pass':<6} Summary")
    print("=" * 110)

    for r in results:
        icon = "✅" if r.passed else "❌"
        print(f"{r.case_id:<20} {r.smell_type:<8} {r.expected_verdict:<16} {r.actual_verdict:<16} {icon:<6} {r.summary[:60]}...")

    print("=" * 110)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    accuracy = passed / total * 100 if total else 0.0
    print(f"\nTotal: {total} | Passed: {passed} | Failed: {total - passed} | Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Use SLM‑1 to generate refactorings (slow). Default uses pre‑defined refactorings.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON file to save detailed results.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N cases.")
    args = parser.parse_args()

    print("Running evaluation...")
    results = run_evaluation(use_slm_generation=args.generate, limit=args.limit)

    print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)
        print(f"Results saved to {args.output}")
