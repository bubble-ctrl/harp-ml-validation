"""
utils/ast_helpers.py — Shared AST utilities for both GNC and Merge validators.
"""

import ast
import pandas as pd


_VALIDATE_ALIASES = {"1:1": "one_to_one", "1:m": "one_to_many",
                     "m:1": "many_to_one", "m:m": "many_to_many"}


def extract_merge_spec(code: str) -> dict:
    """Extract literal merge parameters for structural contract comparison."""
    tree = ast.parse(code)
    call = next((node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "merge"), None)
    if call is None:
        raise ValueError("No DataFrame.merge call found.")
    values = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        try:
            values[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, TypeError):
            values[kw.arg] = None
    if "on" in values:
        values["left_on"] = values["right_on"] = values["on"]
    values["how"] = values.get("how", "inner")
    values["validate"] = _VALIDATE_ALIASES.get(values.get("validate"), values.get("validate"))
    values["explicit_on"] = "on" in values or ("left_on" in values and "right_on" in values)
    values["explicit_how"] = "how" in {kw.arg for kw in call.keywords}
    values["explicit_validate"] = "validate" in values
    return values


def extract_function(code: str, func_name: str = None) -> callable:
    """Extract the first function from a code string and execute it."""
    if func_name is None:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                break
    if func_name is None:
        raise ValueError("No function found in code snippet.")

    namespace = {}
    exec(code, namespace)
    if func_name not in namespace:
        raise ValueError(f"Function {func_name} not found.")
    return namespace[func_name]


def recover_schema(code: str, var_name: str) -> list[str]:
    """
    Recover DataFrame column names from pd.DataFrame({...}) assignment.
    Handles both:
      - pd.DataFrame(data={'col1': [...], 'col2': [...]})
      - pd.DataFrame({'col1': [...], 'col2': [...]})
    """
    tree = ast.parse(code)
    columns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    if isinstance(node.value, ast.Call):
                        func = node.value.func
                        if isinstance(func, ast.Attribute) and func.attr == "DataFrame":
                            # 1. Try keyword argument 'data'
                            for kw in node.value.keywords:
                                if kw.arg == "data" and isinstance(kw.value, ast.Dict):
                                    for key in kw.value.keys:
                                        if isinstance(key, ast.Constant):
                                            columns.append(str(key.value))
                                        elif isinstance(key, ast.Str):
                                            columns.append(key.s)
                            # 2. If no keyword found, check positional args
                            if not columns and node.value.args:
                                first_arg = node.value.args[0]
                                if isinstance(first_arg, ast.Dict):
                                    for key in first_arg.keys:
                                        if isinstance(key, ast.Constant):
                                            columns.append(str(key.value))
                                        elif isinstance(key, ast.Str):
                                            columns.append(key.s)
    return columns

def infer_left_right_vars(code: str) -> tuple[str, str]:
    """Find the two DataFrames in a merge call."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "merge":
                right = node.args[0].id if node.args and isinstance(node.args[0], ast.Name) else "df2"
                left = node.func.value.id if isinstance(node.func.value, ast.Name) else "df1"
                return left, right
    return "df1", "df2"


def structural_merge_check(code: str) -> bool:
    """
    Return True if the merge is already correctly specified (NO_SMELL).
    Requires explicit on (or left_on/right_on) AND how AND validate.
    Implicit on is NOT accepted.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "merge":
                kwargs = {kw.arg: kw.value for kw in node.keywords}
                seen_on = any(k in kwargs for k in ("on", "left_on", "right_on"))
                seen_how = "how" in kwargs
                seen_validate = "validate" in kwargs
                seen_index = (
                    isinstance(kwargs.get("left_index"), ast.Constant)
                    and kwargs["left_index"].value is True
                    and isinstance(kwargs.get("right_index"), ast.Constant)
                    and kwargs["right_index"].value is True
                )

                # Index merges are valid (explicit)
                if seen_index:
                    return True

                # Require all three parameters explicitly
                if seen_on and seen_how and seen_validate:
                    return True

                # If any are missing, it's a smell
                return False
    return False
