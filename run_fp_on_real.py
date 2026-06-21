#!/usr/bin/env python3
"""
run_fp_on_real.py — Run Stage 1 false‑positive filter on real NICHE repositories.

Expects a CSV with columns:
    filename    : Windows absolute path (e.g., F:\projects\cherry\...)
    line        : line number where the smell is reported
    smell_name  : e.g., 'gradients_not_cleared_before_backward_propagation'
    github_repo : e.g., 'learnables/cherry' (used to map to local clone)

Uses the champion set CSV from CodeSmile validation.
"""

import sys
import os
import ast
import textwrap
import pandas as pd
from pathlib import Path

# Add project root to path so we can import our modules
sys.path.insert(0, '.')

from validators.gnc.stage1_intent import check_intentionality
from utils.ast_helpers import structural_merge_check

# =====================================================================
#  CONFIGURATION – CHANGE THESE
# =====================================================================

# Path to the CSV file with smell instances
CSV_PATH = "champion_set_gradients_not_cleared_before_backward_propagation.csv"
# If you want to test Merge, use champion_set_merge_api_parameter_not_explicitly_set.csv

# Base directory where repositories are cloned
REPO_BASE = "./cloned_repos"

# List of repositories to test (use the "github_repo" strings as in the CSV)
SELECTED_REPOS = [
    "pytorch/botorch",
    "huggingface/transformers",
    "ultralytics/yolov5"
]

# Set to True if you want to test Merge smells instead of GNC
TEST_MERGE = False   # Change to True for Merge

# =====================================================================
#  UTILITY FUNCTIONS
# =====================================================================

def win_path_to_relative(win_path: str, repo_folder: str) -> str:
    """
    Convert a Windows absolute path (e.g., F:\projects\cherry\cherry\...)
    to a relative path from the repo root (e.g., cherry/...).
    We look for the repo_folder name (case‑insensitive) and take the rest.
    """
    # Normalize backslashes to forward slashes
    path = win_path.replace("\\", "/")
    parts = path.split("/")
    repo_lower = repo_folder.lower()
    for i, part in enumerate(parts):
        # If the part contains the repo folder (ignoring case), take the rest
        if repo_lower in part.lower():
            return "/".join(parts[i+1:])
    # Fallback: look for common top‑level directories
    common_dirs = ["test", "examples", "utils", "benchmarks", "src", "cherry", "egg", "torchani"]
    for i, part in enumerate(parts):
        if part.lower() in common_dirs:
            return "/".join(parts[i:])
    # Last resort: return the filename only
    return parts[-1]


def extract_function_containing_line(file_path: Path, line_num: int) -> str | None:
    """
    Extract the full function definition that contains the given line number.
    Uses AST first, falls back to indentation‑based extraction.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
    except Exception:
        return None

    # Try AST first
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    if node.lineno <= line_num <= node.end_lineno:
                        lines = source.splitlines()
                        start = node.lineno - 1
                        end = node.end_lineno - 1
                        return '\n'.join(lines[start:end+1])
    except SyntaxError:
        pass  # Fall back to indentation method

    # Indentation‑based fallback
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return None

    # Find the def line
    def_idx = None
    for i in range(line_num-1, max(0, line_num-30)-1, -1):
        if i < len(lines) and lines[i].strip().startswith("def "):
            def_idx = i
            break
    if def_idx is None:
        return None

    # Determine indentation of def
    def_indent = len(lines[def_idx]) - len(lines[def_idx].lstrip())
    body = []
    for i in range(def_idx, len(lines)):
        line = lines[i]
        if i > def_idx and line.strip():
            indent = len(line) - len(line.lstrip())
            if indent <= def_indent and not line.strip().startswith(('@', ')')):
                if line.strip().startswith(('def ', 'class ')):
                    break
                if indent < def_indent:
                    break
        body.append(line)
    return ''.join(body)


def normalize_snippet(code: str) -> str:
    """
    Ensure the extracted code is properly indented and can be parsed.
    """
    return textwrap.dedent(code).strip()


# =====================================================================
#  MAIN
# =====================================================================

def main():
    # Read CSV
    df = pd.read_csv(CSV_PATH)

    # Determine which smell type we're testing
    if TEST_MERGE:
        smell_keyword = "merge_api_parameter_not_explicitly_set"
        df_smell = df[df['smell_name'].str.contains("merge", case=False)]
    else:
        smell_keyword = "gradients_not_cleared"
        df_smell = df[df['smell_name'].str.contains("gradients", case=False)]

    # Filter by selected repos
    df_selected = df_smell[df_smell['github_repo'].isin(SELECTED_REPOS)]
    print(f"Found {len(df_selected)} instances in selected repos for smell: {smell_keyword}")

    results = []

    for idx, row in df_selected.iterrows():
        repo = row['github_repo']
        win_path = row['filename']
        line = row['line']

        repo_folder = repo.split('/')[-1]   # e.g., "cherry"
        repo_path = Path(REPO_BASE) / repo_folder

        rel_path = win_path_to_relative(win_path, repo_folder)
        local_file_path = repo_path / rel_path

        if not local_file_path.exists():
            # Try alternative: sometimes the relative path already includes the repo name
            alt_path = repo_path / Path(win_path).name
            if alt_path.exists():
                local_file_path = alt_path
            else:
                print(f"⚠️ File not found: {local_file_path}")
                continue

        # Extract function containing the line
        snippet = extract_function_containing_line(local_file_path, line)
        if snippet is None:
            print(f"⚠️ Could not extract function from {local_file_path} at line {line}")
            continue

        snippet = normalize_snippet(snippet)

        # Run the appropriate filter
        if TEST_MERGE:
            # For Merge, use the structural check
            is_explicit = structural_merge_check(snippet)
            verdict = 'NO_SMELL' if is_explicit else 'GENUINE'
            reason = "structural check" if is_explicit else "merge parameters missing"
            confidence = 1.0
        else:
            # GNC: use Stage 1 intentionality checker
            result = check_intentionality(snippet)
            verdict = result.verdict.value
            reason = result.reason
            confidence = result.confidence

        results.append({
            "repo": repo,
            "file": str(local_file_path),
            "line": line,
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
        })
        print(f"{repo} | line {line} | {verdict} | {reason[:60]}...")

    # Save results
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv("fp_results_real.csv", index=False)
        print(f"\n✅ Results saved to fp_results_real.csv")
        print(f"Verdict distribution:\n{df_results['verdict'].value_counts()}")
    else:
        print("No instances could be processed.")


if __name__ == "__main__":
    main()