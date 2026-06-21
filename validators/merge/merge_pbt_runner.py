"""Property-based behavioral checks for a single Pandas merge operation."""

from __future__ import annotations

import ast
from typing import Any, Callable

import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings

from utils.ast_helpers import infer_left_right_vars, recover_schema


def _as_keys(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def extract_merge_callable(code: str) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Compile only the candidate merge expression with injectable operands."""
    tree = ast.parse(code)
    call = next((node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "merge"), None)
    if call is None:
        raise ValueError("No DataFrame.merge call found in candidate code.")
    safe_call = ast.Call(
        func=ast.Attribute(value=ast.Name(id="left_df", ctx=ast.Load()), attr="merge", ctx=ast.Load()),
        args=[ast.Name(id="right_df", ctx=ast.Load())],
        keywords=call.keywords,
    )
    expression = ast.Expression(body=safe_call)
    ast.fix_missing_locations(expression)
    compiled = compile(expression, "<merge-candidate>", "eval")

    def merge(left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
        return eval(compiled, {"pd": pd, "np": np}, {"left_df": left_df, "right_df": right_df})

    return merge


def _key_sequences(validate: str | None, size: int, offset: int) -> tuple[list[int], list[int]]:
    base = list(range(offset, offset + size))
    if validate == "one_to_many":
        return base, [value for value in base for _ in range(2)]
    if validate == "many_to_one":
        return [value for value in base for _ in range(2)], base
    if validate == "many_to_many":
        repeated = [value for value in base for _ in range(2)]
        return repeated, list(repeated)
    return base, list(base)


def _make_frames(original_code: str, contract: dict, size: int, offset: int,
                 disjoint: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    left_var, right_var = infer_left_right_vars(original_code)
    left_cols = recover_schema(original_code, left_var)
    right_cols = recover_schema(original_code, right_var)
    left_keys = _as_keys(contract.get("left_on") or contract.get("on_key"))
    right_keys = _as_keys(contract.get("right_on") or contract.get("on_key"))
    if not left_keys or not right_keys or len(left_keys) != len(right_keys):
        raise ValueError("Contract must provide matching left_on/right_on keys.")
    for key in left_keys:
        if key not in left_cols:
            left_cols.append(key)
    for key in right_keys:
        if key not in right_cols:
            right_cols.append(key)

    left_values, right_values = _key_sequences(contract.get("validate"), size, offset)
    if disjoint:
        right_values = [value + 10_000 for value in right_values]
    left_data = {col: [float(i) for i in range(len(left_values))] for col in left_cols}
    right_data = {col: [float(i + 100) for i in range(len(right_values))] for col in right_cols}
    for index, key in enumerate(left_keys):
        left_data[key] = [value * 10 + index for value in left_values]
    for index, key in enumerate(right_keys):
        right_data[key] = [value * 10 + index for value in right_values]
    left_data["_harp_left_id"] = list(range(len(left_values)))
    right_data["_harp_right_id"] = list(range(len(right_values)))
    return pd.DataFrame(left_data), pd.DataFrame(right_data)


def contract_check(contract: dict, left_df: pd.DataFrame, right_df: pd.DataFrame,
                   result_df: pd.DataFrame) -> bool:
    """Check preservation and cardinality-aware row invariants."""
    if not isinstance(result_df, pd.DataFrame):
        return False
    left_ids = set(result_df.get("_harp_left_id", pd.Series(dtype=float)).dropna())
    right_ids = set(result_df.get("_harp_right_id", pd.Series(dtype=float)).dropna())
    if contract.get("left_rows_preserved") and left_ids != set(left_df["_harp_left_id"]):
        return False
    if contract.get("right_rows_preserved") and right_ids != set(right_df["_harp_right_id"]):
        return False
    invariant = contract.get("row_count_invariant")
    if invariant == "eq_left" and len(result_df) != len(left_df):
        return False
    if invariant == "lte_min" and len(result_df) > min(len(left_df), len(right_df)):
        return False
    if invariant == "gte_max" and len(result_df) < max(len(left_df), len(right_df)):
        return False
    return True


def run_pbt(merged_function: Callable, contract: dict, original_code: str,
            runs: int = 100) -> tuple[bool, str]:
    """Exercise the candidate with generated inputs satisfying its preconditions."""

    @given(size=st.integers(min_value=1, max_value=8),
           offset=st.integers(min_value=-100, max_value=100),
           disjoint=st.booleans())
    @settings(max_examples=runs, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def property_test(size: int, offset: int, disjoint: bool) -> None:
        left_df, right_df = _make_frames(original_code, contract, size, offset, disjoint)
        result = merged_function(left_df, right_df)
        assert contract_check(contract, left_df, right_df, result), "behavioral contract violated"

    try:
        property_test()
        return True, "All generated examples satisfied the independent contract."
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
