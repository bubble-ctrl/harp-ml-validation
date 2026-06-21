"""Deterministic, candidate-independent fallback for Merge contracts.

The fallback reads only the original program.  It must never derive the desired
contract from the proposed refactoring: doing so would make validation circular.
"""

from __future__ import annotations

import ast
from difflib import SequenceMatcher
from typing import Any, Optional


VALIDATE_ALIASES = {
    "1:1": "one_to_one",
    "1:m": "one_to_many",
    "m:1": "many_to_one",
    "m:m": "many_to_many",
}


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None


def _dataframes(tree: ast.AST) -> dict[str, dict[Any, list[Any]]]:
    frames: dict[str, dict[Any, list[Any]]] = {}
    column_sets: dict[str, list[Any]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        target = next((t.id for t in node.targets if isinstance(t, ast.Name)), None)
        if (target and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "from_tuples" and node.value.args):
            tuples = _literal(node.value.args[0])
            if isinstance(tuples, list):
                column_sets[target] = tuples
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Attribute) or node.value.func.attr != "DataFrame":
            continue
        target = next((t.id for t in node.targets if isinstance(t, ast.Name)), None)
        if target is None:
            continue
        data_node = node.value.args[0] if node.value.args else next(
            (kw.value for kw in node.value.keywords if kw.arg == "data"), None
        )
        data = _literal(data_node) if data_node is not None else None
        if isinstance(data, dict):
            frames[target] = {key: list(value) for key, value in data.items() if isinstance(value, (list, tuple))}
            continue
        columns_node = next((kw.value for kw in node.value.keywords if kw.arg == "columns"), None)
        columns = column_sets.get(columns_node.id, []) if isinstance(columns_node, ast.Name) else _literal(columns_node)
        rows = _literal(data_node) if data_node is not None else None
        if isinstance(columns, list) and isinstance(rows, list) and rows:
            frames[target] = {
                col: [row[index] for row in rows if isinstance(row, (list, tuple)) and len(row) > index]
                for index, col in enumerate(columns)
            }
    return frames


def _merge_call(tree: ast.AST) -> Optional[ast.Call]:
    return next((node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "merge"), None)


def _name(node: ast.AST) -> Optional[str]:
    return node.id if isinstance(node, ast.Name) else None


def _kw(call: ast.Call, name: str) -> Any:
    node = next((kw.value for kw in call.keywords if kw.arg == name), None)
    return _literal(node) if node is not None else None


def _best_keys(left: dict, right: dict) -> tuple[Any, Any]:
    if not left or not right:
        return None, None
    candidates: list[tuple[float, Any, Any, float]] = []
    for left_key, left_values in left.items():
        for right_key, right_values in right.items():
            left_set, right_set = set(left_values), set(right_values)
            overlap = len(left_set & right_set) / max(1, min(len(left_set), len(right_set)))
            name_score = SequenceMatcher(None, str(left_key), str(right_key)).ratio()
            score = 2.0 * overlap + name_score + (0.5 if left_key == right_key else 0.0)
            candidates.append((score, left_key, right_key, overlap))
    candidates.sort(reverse=True, key=lambda item: item[0])
    chosen_left, chosen_right = [], []
    for _, left_key, right_key, overlap in candidates:
        if overlap < 0.8 or left_key in chosen_left or right_key in chosen_right:
            continue
        chosen_left.append(left_key)
        chosen_right.append(right_key)
    if not chosen_left:
        return candidates[0][1], candidates[0][2]
    if len(chosen_left) == 1:
        return chosen_left[0], chosen_right[0]
    return chosen_left, chosen_right


def _cardinality(left_values: list[Any], right_values: list[Any]) -> Optional[str]:
    if not left_values or not right_values:
        return None
    left_unique = len(left_values) == len(set(left_values))
    right_unique = len(right_values) == len(set(right_values))
    if left_unique and right_unique:
        return "one_to_one"
    if left_unique:
        return "one_to_many"
    if right_unique:
        return "many_to_one"
    return "many_to_many"


def _intended_how(code: str, explicit: Optional[str]) -> tuple[str, str]:
    if explicit:
        return explicit, "explicit"
    text = code.lower()
    if any(term in text for term in ("full outer", "union of", "both sides", "track left/right",
                                     "fill_missing", "with_indicator")):
        return "outer", "context"
    if any(term in text for term in ("preserve all", "retain all", "drops customers", "self_join", "self-join")):
        return "left", "context"
    return "inner", "default"


def infer_contract_from_ast(original_code: str, refactored_code: str = "") -> Optional[dict]:
    """Infer a conservative desired contract using *only* ``original_code``.

    ``refactored_code`` remains in the signature for API compatibility and is
    deliberately ignored.
    """
    try:
        tree = ast.parse(original_code)
    except SyntaxError:
        return None
    call = _merge_call(tree)
    if call is None:
        return None

    frames = _dataframes(tree)
    left_var = _name(call.func.value)
    right_var = _name(call.args[0]) if call.args else None
    left_data = frames.get(left_var or "", {})
    right_data = frames.get(right_var or "", {})

    on = _kw(call, "on")
    left_on = _kw(call, "left_on")
    right_on = _kw(call, "right_on")
    if on is not None:
        left_on = right_on = on
    if left_on is None or right_on is None:
        inferred_left, inferred_right = _best_keys(left_data, right_data)
        left_on = left_on if left_on is not None else inferred_left
        right_on = right_on if right_on is not None else inferred_right

    left_keys = left_on if isinstance(left_on, list) else [left_on]
    right_keys = right_on if isinstance(right_on, list) else [right_on]
    left_values = left_data.get(left_keys[0], []) if left_keys and left_keys[0] is not None else []
    right_values = right_data.get(right_keys[0], []) if right_keys and right_keys[0] is not None else []

    explicit_validate = _kw(call, "validate")
    validate = VALIDATE_ALIASES.get(explicit_validate, explicit_validate)
    validate = validate or _cardinality(left_values, right_values)
    how, how_source = _intended_how(original_code, _kw(call, "how"))

    left_nonkeys = set(left_data) - set(left_keys)
    right_nonkeys = set(right_data) - set(right_keys)
    suffixes_required = bool(left_nonkeys & right_nonkeys) or left_var == right_var
    if how == "left" and validate in {"one_to_one", "many_to_one"}:
        row_invariant = "eq_left"
    elif how == "outer":
        row_invariant = "gte_max"
    elif how == "inner" and validate == "one_to_one":
        row_invariant = "lte_min"
    else:
        row_invariant = "unknown"

    return {
        "how": how,
        "how_source": how_source,
        "row_count_invariant": row_invariant,
        "left_rows_preserved": how in {"left", "outer"},
        "right_rows_preserved": how in {"right", "outer"},
        "on_key": left_on if left_on == right_on else None,
        "left_on": left_on,
        "right_on": right_on,
        "validate": validate,
        "suffixes_required": suffixes_required,
        "confidence": "medium",
        "reasoning": "Deterministic fallback inferred from the original merge, literal schemas, values, and context.",
    }
