"""
utils/ast_helpers.py — Shared AST utilities for both GNC and Merge validators.
"""

import ast
import pandas as pd


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
                            for kw in node.value.keywords:
                                if kw.arg == "data":
                                    if isinstance(kw.value, ast.Dict):
                                        for key in kw.value.keys:
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
    """Return True if merge already has on, how, and validate."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "merge":
                seen_on = seen_how = seen_validate = False
                for kw in node.keywords:
                    if kw.arg in ("on", "left_on", "right_on"):
                        seen_on = True
                    elif kw.arg == "how":
                        seen_how = True
                    elif kw.arg == "validate":
                        seen_validate = True
                if seen_on and seen_how and seen_validate:
                    return True
    return False