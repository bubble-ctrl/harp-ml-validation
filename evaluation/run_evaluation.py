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


@dataclass
class EvalResult:
    case_id: str
    smell_type: str
    expected_verdict: str
    actual_verdict: str
    passed: bool
    summary: str


def run_evaluation(use_slm_generation: bool = False) -> List[EvalResult]:
    """
    Run all test cases.
    If use_slm_generation is True, SLM‑1 generates the refactoring from scratch.
    Otherwise, the pre‑defined after_correct / after_wrong are used.
    """
    results = []

    for case_id, case in TEST_CASES.items():
        smell_type = case.get("smell_type")
        before = case["before"]
        expected = case["expected_validator"]

        # Determine which refactored code to use
        if use_slm_generation:
            # Call SLM‑1 to generate refactoring
            if smell_type == "gnc":
                from models.slm_refactor import refactor_gnc
                refactor_result = refactor_gnc(before)
                if not refactor_result.success:
                    print(f"[{case_id}] SLM‑1 failed: {refactor_result.error}")
                    continue
                refactored = refactor_result.refactored_code
                # SLM‑1 might return false positive
                if refactor_result.is_false_positive:
                    actual = "NO_SMELL" if smell_type == "gnc" else "NO_SMELL"
                    passed = actual == expected
                    results.append(EvalResult(case_id, smell_type, expected, actual, passed, "SLM‑1 flagged FP"))
                    continue
            else:  # merge
                from models.slm_refactor import refactor_merge
                refactor_result = refactor_merge(before)
                if not refactor_result.success:
                    print(f"[{case_id}] SLM‑1 failed: {refactor_result.error}")
                    continue
                refactored = refactor_result.refactored_code
        else:
            # Use the pre‑defined "correct" refactoring
            refactored = case.get("after_correct")
            if not refactored:
                # If no after_correct, it's a false positive case; validator should short‑circuit
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
    print(f"\nTotal: {total} | Passed: {passed} | Failed: {total - passed} | Accuracy: {passed/total*100:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Use SLM‑1 to generate refactorings (slow). Default uses pre‑defined refactorings.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON file to save detailed results.")
    args = parser.parse_args()

    print("Running evaluation...")
    results = run_evaluation(use_slm_generation=args.generate)

    print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)
        print(f"Results saved to {args.output}")