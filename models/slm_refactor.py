"""
slm_refactor.py — SLM‑1 refactoring for GNC and Merge API smells.

Uses the shared model loader (configurable via config.py).
"""

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
# Prompt builders (using the templates you provided)
# --------------------------------------------------------------------

def build_gnc_prompt(code_snippet: str, start_line: int, end_line: int,
                     smell_line: int, relative_line: int) -> str:
    return f"""You are a PyTorch code refactoring expert.

**Code Smell**: Missing optimizer.zero_grad() before loss.backward()

**Known False Positive Patterns** — do NOT refactor if any apply:
- optimizer.zero_grad() is already present at the top of the training loop,
  even if separated from loss.backward() by intervening statements such as
  the forward pass and loss computation.
- Gradient accumulation is architecturally intentional, controlled by a
  runtime condition or a nested minibatch loop within a custom backward hook.
- zero_grad() is called on a different optimizer instance earlier in the loop,
  making accumulation intentional for multi-optimizer training.

**Correct Pattern**:
```python
for batch in dataloader:
    optimizer.zero_grad()  # Clear gradients
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
```

Code to Fix (lines {start_line} to {end_line}):

```python
{code_snippet}
```
Smell is at line {smell_line} (line {relative_line} of the snippet above).

Provide:

REASONING: One sentence explaining the fix and why the smell is harmful.

REFACTORED CODE: Only the fixed code section (the window above with fix applied).

If this is a false positive, return the original code unchanged and explain
why in REASONING.
"""

def build_merge_prompt(code_snippet: str, start_line: int, end_line: int,
                       smell_line: int, relative_line: int) -> str:
    return f"""You are a Python code refactoring expert specializing in Pandas.

**Code Smell**: Merge API Parameter Not Explicitly Set

A df.merge() call is missing one or more of the following parameters
that should be explicitly specified for clarity, correctness, and readability:
  - `on`       : which column(s) to join on
  - `how`      : join method (inner, outer, left, right)
  - `validate` : checks merge key uniqueness (e.g. "one_to_one", "one_to_many")

**Why this matters**:
  - Without `on`, Pandas silently merges on all common column names, which
    can produce incorrect results if schemas change.
  - Without `how`, the default is inner join, which may silently drop rows.
  - Without `validate`, duplicate keys can cause silent data duplication.

**Known False Positive Patterns** — return FALSE POSITIVE if any apply:
  - `on`, `how`, and `validate` are already all explicitly specified.
  - The merge is on a single unambiguous index with no common columns.

**Correct Pattern**:
```python
# Before (smell): parameters not explicit
result = df1.merge(df2)

# After (fix): parameters explicitly specified
result = df1.merge(
    df2,
    on="user_id",
    how="inner",
    validate="many_to_one"
)
```

Code Window (lines {start_line} to {end_line}):

```python
{code_snippet}
```
Smell is at line {smell_line} (line {relative_line} of the snippet above).

Use the surrounding context to infer the correct values for on, how,
and validate. If you cannot determine the correct value with confidence,
use None as a placeholder.

You MUST respond in exactly one of these two formats:

Format 1 — if this is a false positive:
FALSE POSITIVE: <one sentence explaining why>

Format 2 — if refactoring is needed:
REASONING: <one sentence explaining what parameters were added and why>
REFACTORED CODE:

```python
<only the fixed code window>
```
"""

# --------------------------------------------------------------------
# Model inference
# --------------------------------------------------------------------

def query_model(prompt: str, max_new_tokens: int = 128) -> str:   # reduced from 256 to 128
    """Run locally (CPU) with reduced tokens for faster generation."""
    print("[SLM‑1] Using local CPU (may take 30-90 seconds).")
    return _query_local(prompt, max_new_tokens)


def _query_local(prompt: str, max_new_tokens: int) -> str:
    """Run model locally on CPU/GPU."""
    model, tokenizer = get_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in response:
        response = response.split(prompt)[-1].strip()
    return response
# --------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------
def parse_gnc_response(response: str, original_code: str) -> RefactorResult:
    # Check for false positive
    if "FALSE POSITIVE" in response.upper():
        match = re.search(r"FALSE POSITIVE:\s*(.+)", response, re.IGNORECASE)
        reason = match.group(1).strip() if match else "False positive"
        return RefactorResult(
            success=True,
            refactored_code=original_code,
            reasoning=reason,
            is_false_positive=True
        )

    # Try to extract reasoning
    reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Refactored code."

    # Try to extract code block – first with triple backticks
    code_match = re.search(r"REFACTORED CODE:\s*```python\s*(.*?)\s*```", response, re.IGNORECASE | re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        return RefactorResult(success=True, refactored_code=code, reasoning=reasoning, is_false_positive=False)

    # Fallback: look for code after "REFACTORED CODE:" without backticks
    code_match2 = re.search(r"REFACTORED CODE:\s*(.*?)(?:\n\s*\n|$)", response, re.IGNORECASE | re.DOTALL)
    if code_match2:
        code = code_match2.group(1).strip()
        # If it looks like Python code (contains def, for, etc.), accept it
        if code and ("def " in code or "for " in code or "optimizer.zero_grad" in code):
            return RefactorResult(success=True, refactored_code=code, reasoning=reasoning, is_false_positive=False)

    # If still nothing, try to find a code block without any marker
    # (sometimes models just output the code directly)
    code_block = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    if code_block:
        code = code_block.group(1).strip()
        return RefactorResult(success=True, refactored_code=code, reasoning=reasoning, is_false_positive=False)

    # Last resort: try to extract any indented block that looks like code
    lines = response.splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith("for ") or line.strip().startswith("def ") or line.strip().startswith("optimizer.zero_grad"):
            in_code = True
        if in_code:
            code_lines.append(line)
            if line.strip() and not line.startswith(" "):  # stop when no indent
                # but keep going until empty line
                pass
    if code_lines:
        code = "\n".join(code_lines).strip()
        if code:
            return RefactorResult(success=True, refactored_code=code, reasoning=reasoning, is_false_positive=False)

    # No code found
    return RefactorResult(
        success=False,
        refactored_code="",
        reasoning=reasoning,
        is_false_positive=False,
        error=f"Could not parse GNC response: {response[:200]}..."
    )

def parse_merge_response(response: str, original_code: str) -> RefactorResult:
    """
    Parse the model's response for Merge API refactoring.
    Expected formats:
      - "FALSE POSITIVE: <reason>"
      - "REASONING: ...\nREFACTORED CODE:\n```python\n...```"
    """
    # 1. Check for false positive
    if "FALSE POSITIVE" in response.upper():
        match = re.search(r"FALSE POSITIVE:\s*(.+)", response, re.IGNORECASE)
        reason = match.group(1).strip() if match else "False positive"
        return RefactorResult(
            success=True,
            refactored_code=original_code,
            reasoning=reason,
            is_false_positive=True
        )

    # 2. Extract reasoning
    reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Refactored merge parameters added."

    # 3. Try to extract code block – triple backticks
    code_match = re.search(r"REFACTORED CODE:\s*```python\s*(.*?)\s*```", response, re.IGNORECASE | re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        if code:
            return RefactorResult(
                success=True,
                refactored_code=code,
                reasoning=reasoning,
                is_false_positive=False
            )

    # 4. Fallback: look for code after "REFACTORED CODE:" without backticks
    code_match2 = re.search(r"REFACTORED CODE:\s*(.*?)(?:\n\s*\n|$)", response, re.IGNORECASE | re.DOTALL)
    if code_match2:
        code = code_match2.group(1).strip()
        if code and ("df" in code or "merge" in code or "on=" in code):
            return RefactorResult(
                success=True,
                refactored_code=code,
                reasoning=reasoning,
                is_false_positive=False
            )

    # 5. Try to find a Python code block without the "REFACTORED CODE" marker
    code_block = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    if code_block:
        code = code_block.group(1).strip()
        if code:
            return RefactorResult(
                success=True,
                refactored_code=code,
                reasoning=reasoning,
                is_false_positive=False
            )

    # 6. Last resort: scan for a line containing ".merge("
    lines = response.splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") or "merge" in stripped or "df" in stripped:
            in_code = True
        if in_code:
            # If we hit an empty line after starting code, we might still continue
            if stripped == "" and code_lines and code_lines[-1].strip() == "":
                # avoid double empty lines
                pass
            code_lines.append(line)
            # Stop if we see "FALSE POSITIVE" or "REASONING" again (unlikely)
            if "FALSE POSITIVE" in stripped.upper() or "REASONING" in stripped.upper():
                break

    if code_lines:
        code = "\n".join(code_lines).strip()
        # Ensure it actually contains some Pandas/Python code
        if code and any(keyword in code for keyword in ["df.", "merge", "pd.", "on=", "how="]):
            return RefactorResult(
                success=True,
                refactored_code=code,
                reasoning=reasoning,
                is_false_positive=False
            )

    # No code found
    return RefactorResult(
        success=False,
        refactored_code="",
        reasoning=reasoning,
        is_false_positive=False,
        error=f"Could not parse Merge response: {response[:200]}..."
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

