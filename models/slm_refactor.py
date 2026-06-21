"""
slm_refactor.py — Robust SLM‑1 refactoring for GNC and Merge API smells.

Optimized for DeepSeek-Coder-6.7B-Instruct with fallback heuristic processors
to handle tokenization collapse (stripped spaces) or placeholder hallucinations.
"""

import ast
import re
import torch
from dataclasses import dataclass
from typing import Optional
from models.shared_model import get_model   # ← use shared loader

# --------------------------------------------------------------------
# Dataclass matching pipeline.py expectations
# --------------------------------------------------------------------

@dataclass
class RefactorResult:
    success: bool
    refactored_code: str
    reasoning: str
    is_false_positive: bool = False
    error: Optional[str] = None

# --------------------------------------------------------------------
# Programmatic Failsafe Heuristics (Robust Fallbacks)
# --------------------------------------------------------------------

def heuristic_fix_gnc(code: str) -> str:
    """Fallback compiler to cleanly inject zero_grad if the SLM loses spacing."""
    lines = code.splitlines()
    fixed_lines = []
    
    # Look for existing zero_grad
    if any("zero_grad" in line for line in lines):
        return code

    # Discover the optimizer instance name dynamically
    optimizer_name = "optimizer"
    for line in lines:
        if ".step()" in line:
            match = re.search(r"(\w+)\.step\(\)", line)
            if match:
                optimizer_name = match.group(1)
                break

    inserted = False
    for line in lines:
        if "backward()" in line and not inserted:
            indent = len(line) - len(line.lstrip())
            fixed_lines.append(" " * indent + f"{optimizer_name}.zero_grad()")
            inserted = True
        fixed_lines.append(line)
        
    return "\n".join(fixed_lines)


def heuristic_fix_merge(code: str) -> str:
    """Deprecated: guessing merge semantics is unsafe for a Type-II smell."""
    return code


_MERGE_KEYWORDS = {
    "on", "left_on", "right_on", "left_index", "right_index", "how",
    "validate", "suffixes", "sort", "indicator", "copy",
}


def _first_merge(tree: ast.AST) -> Optional[ast.Call]:
    return next((node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "merge"), None)


def _extract_python_candidate(response: str) -> Optional[str]:
    """Return the first parseable Python candidate from an SLM response."""
    candidates = re.findall(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL | re.IGNORECASE)
    candidates.append(response.strip())
    lines = response.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            block = [line]
            for following in lines[i + 1:]:
                if not following.strip() or following[:1].isspace():
                    block.append(following)
                else:
                    break
            candidates.append("\n".join(block))
    ranked = []
    for candidate in candidates:
        try:
            tree = ast.parse(candidate)
            merge = _first_merge(tree)
            if merge is None:
                continue
            names = {kw.arg for kw in merge.keywords}
            score = sum(name in names for name in ("on", "left_on", "right_on", "how", "validate"))
            ranked.append((score, candidate))
        except SyntaxError:
            continue
    return max(ranked, key=lambda item: item[0])[1] if ranked else None


def _safe_merge_transplant(original_code: str, candidate_code: str) -> tuple[Optional[str], Optional[str]]:
    """Apply only model-selected merge keyword arguments to the original AST.

    This prevents an otherwise parseable hallucination (for example
    ``pders.DataFrame``) from silently changing unrelated program behavior.
    """
    try:
        original_tree = ast.parse(original_code)
        candidate_tree = ast.parse(candidate_code)
    except SyntaxError as exc:
        return None, f"Generated code is not valid Python: {exc}"
    original_merge = _first_merge(original_tree)
    candidate_merge = _first_merge(candidate_tree)
    if original_merge is None or candidate_merge is None:
        return None, "Generated code does not contain the target merge call."

    candidate_kwargs = {kw.arg: kw for kw in candidate_merge.keywords if kw.arg in _MERGE_KEYWORDS}
    has_keys = "on" in candidate_kwargs or ({"left_on", "right_on"} <= set(candidate_kwargs))
    if not has_keys or "how" not in candidate_kwargs or "validate" not in candidate_kwargs:
        return None, "Generated merge must explicitly set keys, how, and validate."
    for name, keyword in candidate_kwargs.items():
        try:
            ast.literal_eval(keyword.value)
        except (ValueError, TypeError):
            return None, f"Generated {name} must be a literal value."

    existing = {kw.arg: kw for kw in original_merge.keywords if kw.arg is not None}
    if "on" in candidate_kwargs:
        existing.pop("left_on", None)
        existing.pop("right_on", None)
    else:
        existing.pop("on", None)
    existing.update(candidate_kwargs)
    original_merge.keywords = list(existing.values())
    ast.fix_missing_locations(original_tree)
    return ast.unparse(original_tree), None


# --------------------------------------------------------------------
# Prompt builders (No-fluff templates targeted for DeepSeek)
# --------------------------------------------------------------------

def build_gnc_prompt(code_snippet: str, start_line: int, end_line: int,
                     smell_line: int, relative_line: int) -> str:
    return f"""### Instruction:
You are an expert PyTorch refactoring tool. Fix the code smell by adding `optimizer.zero_grad()` directly before `loss.backward()`.

[CRITICAL RULE]
Do not change any variable names or loop conditions. Keep `epoch`, `X`, `y`, `out`, and `loss` exactly as defined.
Write ONLY the refactored code block inside python markdown brackets. No conversational text.

Code window to fix:
```python
{code_snippet}
```

### Response:
```python
"""


def build_merge_prompt(code_snippet: str, start_line: int, end_line: int,
                       smell_line: int, relative_line: int) -> str:
    return f"""### Instruction:
You are an expert Pandas code refactoring tool. Fix the merge statement to explicitly provide `on`, `how`, and `validate` parameters.

[CRITICAL RULE]
Keep original variable assignments.
Write ONLY the refactored code block inside python markdown brackets. No conversational text.

Code window to fix:
```python
{code_snippet}
```

### Response:
```python
"""


# --------------------------------------------------------------------
# Model inference
# --------------------------------------------------------------------

def query_model(prompt: str, max_new_tokens: int = 512, component: str = "SLM-1") -> str:
    """Run locally (CPU/GPU) with structured configurations."""
    print(f"[{component}] Running inference using DeepSeek Engine...")
    return _query_local(prompt, max_new_tokens)


def _query_local(prompt: str, max_new_tokens: int) -> str:
    model, tokenizer = get_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,          # Soft sampling prevents repetitive loop collapse
            temperature=0.1,         # Keeps model output strictly aligned with formatting
            repetition_penalty=1.1,  # Discourages repeating instructions
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in response:
        response = response.split(prompt)[-1].strip()
    return response


# --------------------------------------------------------------------
# Multi-Layer Parsers
# --------------------------------------------------------------------

def parse_gnc_response(response: str, original_code: str) -> RefactorResult:
    # 1. Clean up response headers
    clean_res = response.replace("```python", "").replace("```", "").strip()

    # 2. Check for False Positive signals
    if "FALSE POSITIVE" in clean_res.upper():
        return RefactorResult(
            success=True,
            refactored_code=original_code,
            reasoning="Intentionally configured or gradient accumulation loop present.",
            is_false_positive=True
        )

    # 3. Detect and repair model token space-collapse (e.g., "forepochin") or placeholder hallucinations
    collapsed_whitespace = "forepochin" in clean_res.replace(" ", "")
    contains_placeholder = "oneclearsentence" in clean_res.lower() or "your_code" in clean_res.lower()

    if collapsed_whitespace or contains_placeholder or not clean_res:
        # Trigger heuristic failsafe logic
        fixed = heuristic_fix_gnc(original_code)
        return RefactorResult(
            success=True,
            refactored_code=fixed,
            reasoning="Inserted missing zero_grad() call prior to backward propagation to clear accumulated historical gradients.",
            is_false_positive=False
        )

    # 4. Standard parsed route
    return RefactorResult(
        success=True,
        refactored_code=clean_res,
        reasoning="Successfully injected optimizer.zero_grad() directly before calling backward propagation.",
        is_false_positive=False
    )


def parse_merge_response(response: str, original_code: str) -> RefactorResult:
    # Never execute the model's entire program. Extract its merge decision and
    # transplant only those API keywords onto the original program.
    clean_res = _extract_python_candidate(response)
    print(f"  [DEBUG:parse_merge] Raw response length: {len(response)}")
    print(f"  [DEBUG:parse_merge] Candidate found: {clean_res is not None}")

    # 2. Check for False Positive signals
    if "FALSE POSITIVE" in response.upper():
        return RefactorResult(
            success=True,
            refactored_code=original_code,
            reasoning="Unambiguous simple merge or explicit join patterns are already present.",
            is_false_positive=True
        )

    if clean_res is None:
        return RefactorResult(False, original_code, "", error="SLM output contains no parseable Python code.")
    safe_code, error = _safe_merge_transplant(original_code, clean_res)
    if error:
        return RefactorResult(False, original_code, "", error=error)
    return RefactorResult(
        success=True,
        refactored_code=safe_code or original_code,
        reasoning="Applied only the SLM-selected merge parameters to the unchanged original program.",
        is_false_positive=False
    )


# --------------------------------------------------------------------
# Public refactoring functions
# --------------------------------------------------------------------

def refactor_gnc(code: str) -> RefactorResult:
    lines = code.splitlines()
    start_line = 1
    end_line = len(lines)
    smell_line = 1
    for i, line in enumerate(lines, 1):
        if "loss.backward()" in line or "backward()" in line:
            smell_line = i
            break
    relative_line = smell_line - start_line + 1
    prompt = build_gnc_prompt(code, start_line, end_line, smell_line, relative_line)
    response = query_model(prompt)
    return parse_gnc_response(response, code)


def refactor_merge(code: str) -> RefactorResult:
    lines = code.splitlines()
    start_line = 1
    end_line = len(lines)
    smell_line = 1
    for i, line in enumerate(lines, 1):
        if ".merge(" in line:
            smell_line = i
            break
    relative_line = smell_line - start_line + 1
    prompt = build_merge_prompt(code, start_line, end_line, smell_line, relative_line)
    response = query_model(prompt)
    return parse_merge_response(response, code)


def refactor(smell_type: str, code: str) -> RefactorResult:
    if smell_type.lower() == "gnc":
        return refactor_gnc(code)
    elif smell_type.lower() == "merge":
        return refactor_merge(code)
    else:
        return RefactorResult(
            success=False,
            refactored_code="",
            reasoning="",
            error=f"Unknown smell type: {smell_type}"
        )
