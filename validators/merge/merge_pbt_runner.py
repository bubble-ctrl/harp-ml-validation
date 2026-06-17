"""
validators/merge/merge_pbt_runner.py — Hypothesis PBT runner for Merge API.
"""

import pandas as pd
import hypothesis.strategies as st
from hypothesis import assume, settings, Verbosity
from typing import Callable
from utils.ast_helpers import recover_schema, infer_left_right_vars


def contract_check(contract: dict, left_df: pd.DataFrame, right_df: pd.DataFrame,
                   result_df: pd.DataFrame) -> bool:
    """Check all invariants against the contract."""
    row_inv = contract.get("row_count_invariant")
    if row_inv == "eq_left":
        if len(result_df) != len(left_df):
            return False
    elif row_inv == "lte_min":
        if len(result_df) > min(len(left_df), len(right_df)):
            return False
    elif row_inv == "gte_left":
        if len(result_df) < len(left_df):
            return False

    on_key = contract.get("on_key")
    if contract.get("left_rows_preserved") is True and on_key:
        left_keys = set(left_df[on_key]) if on_key in left_df else set()
        result_keys = set(result_df[on_key]) if on_key in result_df else set()
        if not left_keys.issubset(result_keys):
            return False

    if contract.get("right_rows_preserved") is True and on_key:
        right_keys = set(right_df[on_key]) if on_key in right_df else set()
        result_keys = set(result_df[on_key]) if on_key in result_df else set()
        if not right_keys.issubset(result_keys):
            return False

    validate = contract.get("validate")
    if validate and on_key:
        if validate == "one_to_one":
            if left_df[on_key].duplicated().any() or right_df[on_key].duplicated().any():
                return False
        elif validate == "one_to_many":
            if left_df[on_key].duplicated().any():
                return False
        elif validate == "many_to_one":
            if right_df[on_key].duplicated().any():
                return False
    return True


def build_strategy(left_cols: list, right_cols: list, on_key: str | list | None):
    """Build Hypothesis strategy for left/right DataFrames."""
    if on_key is None:
        on_key = left_cols[0] if left_cols else "key"
    if isinstance(on_key, list):
        on_key = on_key[0]  # simplify for demonstration

    left_strat = st.dataframes(
        columns={
            col: st.integers() if col == on_key else st.floats(allow_nan=False)
            for col in (left_cols if left_cols else ["key", "value"])
        },
        rows=st.integers(min_value=1, max_value=15)
    )
    right_strat = st.dataframes(
        columns={
            col: st.integers() if col == on_key else st.floats(allow_nan=False)
            for col in (right_cols if right_cols else ["key", "value"])
        },
        rows=st.integers(min_value=1, max_value=15)
    )
    return left_strat, right_strat


def run_pbt(merged_function: Callable, contract: dict, original_code: str,
            runs: int = 100) -> tuple[bool, str]:
    """Run Hypothesis PBT and return (passed, message)."""
    left_var, right_var = infer_left_right_vars(original_code)
    left_cols = recover_schema(original_code, left_var)
    right_cols = recover_schema(original_code, right_var)

    if not left_cols or not right_cols:
        left_cols = ["key", "val1"]
        right_cols = ["key", "val2"]

    left_strat, right_strat = build_strategy(left_cols, right_cols, contract.get("on_key"))

    errors = []

    @settings(max_examples=runs, verbosity=Verbosity.quiet, deadline=None)
    def test_contract(left_df: pd.DataFrame, right_df: pd.DataFrame):
        try:
            result = merged_function(left_df, right_df)
            if not contract_check(contract, left_df, right_df, result):
                errors.append("Contract violation")
                assume(False)
        except Exception as e:
            errors.append(f"Exception: {e}")
            assume(False)

    try:
        test_contract()
        if errors:
            return False, f"PBT failed: {errors[0]}"
        return True, "All PBT checks passed."
    except Exception as e:
        return False, f"PBT error: {e}"