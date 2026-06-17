"""
slm2_merge_contract.py — SLM‑2 for Merge API contract inference.
"""

import json
import re
from typing import Optional
from models.shared_model import get_model
from models.slm_refactor import query_model   # re-use inference
# =====================================================================
#  Prompt for SLM‑2 (contract inference)
# =====================================================================

CONTRACT_PROMPT_TEMPLATE = """You are an expert in Pandas and data contracts.

Given the **original code** (which contained a merge smell) and the **refactored code** (which now has explicit merge parameters), infer the **behavioral contract** that the refactored merge must satisfy.

## Behavioral Contract Fields
- `"how"`: "left" | "inner" | "outer" | "right" | "unknown"
- `"row_count_invariant"`: "eq_left" | "lte_min" | "gte_left" | "unknown"
  - "eq_left" -> len(result) == len(left)   (left/right outer join)
  - "lte_min" -> len(result) <= min(len(left), len(right))   (inner join)
  - "gte_left" -> len(result) >= len(left)  (outer join preserving left)
- `"left_rows_preserved"`: true | false   (every left row appears in result)
- `"right_rows_preserved"`: true | false  (every right row appears in result)
- `"on_key"`: "column_name" | ["col1", "col2"] | null   (the join key(s))
- `"validate"`: "one_to_one" | "one_to_many" | "many_to_one" | null
- `"confidence"`: "high" | "medium" | "low"
- `"reasoning"`: short explanation of why you inferred this contract

## Original Code (with smell)
```python
{original_code}
```

## Refactored Code (with explicit merge parameters)
```python
{refactored_code}
```

## Instructions
Use the original code to understand the intent and variable names.

Use the refactored code to see what parameters were added.

If the refactored code uses how='left', then left_rows_preserved MUST be true.

If how='inner', then row_count_invariant is likely "lte_min".

If how='outer', then both left_rows_preserved and right_rows_preserved are true.

If you cannot determine a field with confidence, set it to null or "unknown" and lower confidence.

## Output
Return ONLY a valid JSON object. Do not include any explanation outside the JSON.

json
{{
  "how": "...",
  "row_count_invariant": "...",
  "left_rows_preserved": true,
  "right_rows_preserved": false,
  "on_key": "...",
  "validate": "...",
  "confidence": "...",
  "reasoning": "..."
}}
"""

def infer_contract(original_code: str, refactored_code: str) -> Optional[dict]:
    """
    Call SLM‑2 to infer the behavioral contract.
    Returns a dict with the contract fields, or None on failure.
    """
    prompt = CONTRACT_PROMPT_TEMPLATE.format(
        original_code=original_code,
        refactored_code=refactored_code
    )
    response = query_model(prompt, max_new_tokens=512)

    # Try to extract JSON from the response
    json_match = re.search(r'json\s*(\{.*?\})\s*', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Fallback: find anything that looks like a JSON object
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Last resort: assume the whole response is JSON
            json_str = response

    try:
        contract = json.loads(json_str.strip())

        # Validate required fields
        required = {"how", "row_count_invariant", "left_rows_preserved",
                    "right_rows_preserved", "on_key", "validate", "confidence", "reasoning"}
        if not required.issubset(contract.keys()):
            print(f"[SLM‑2] Missing fields in contract. Got: {contract.keys()}")
            return None
        return contract
    except json.JSONDecodeError as e:
        print(f"[SLM‑2] Failed to parse JSON: {e}")
        print(f"Response was: {response[:500]}...")
        return None
