"""
slm2_merge_contract.py — SLM‑2 for Merge API contract inference.

SLM-2 infers intent from the original program.  A deterministic fallback is
used only when the model cannot return a usable contract.  The proposed
refactoring is intentionally excluded from inference to avoid circular
validation.
"""

import json
import re
from typing import Optional
from models.ast_contract_fallback import infer_contract_from_ast

# Late import to avoid loading torch at import time
_query_model = None

def _get_query_model():
    global _query_model
    if _query_model is None:
        from models.slm_refactor import query_model
        _query_model = query_model
    return _query_model


# =====================================================================
#  Prompt for SLM‑2 (kept for optional use — rarely called)
# =====================================================================

CONTRACT_PROMPT_TEMPLATE = """### Instruction:
You infer the intended behavioral contract of a Pandas merge from the original
program and its comments. Return exactly one valid JSON object and no markdown.

Original code:
```python
{original_code}
```

Required schema:
{{
  "how": "left|inner|outer|right|unknown",
  "row_count_invariant": "eq_left|lte_min|gte_max|unknown",
  "left_rows_preserved": true,
  "right_rows_preserved": false,
  "on_key": "shared_key_or_null",
  "left_on": "left_key_or_list_or_null",
  "right_on": "right_key_or_list_or_null",
  "validate": "one_to_one|one_to_many|many_to_one|many_to_many|null",
  "suffixes_required": false,
  "confidence": "high|medium|low",
  "reasoning": "short evidence-based explanation"
}}

Infer desired behavior, not merely Pandas defaults. Use literal schemas and
duplicate key values as cardinality evidence. Use null/"unknown" when intent is
not supported. Do not invent columns.

### Response:
"""


# =====================================================================
#  Multi-layer JSON extraction (for SLM-2 responses)
# =====================================================================

REQUIRED_FIELDS = {"how", "row_count_invariant", "left_rows_preserved",
                   "right_rows_preserved", "on_key", "validate",
                   "confidence", "reasoning"}


def _try_parse_json(text: str) -> Optional[dict]:
    """Attempt to parse text as JSON and validate required fields."""
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and REQUIRED_FIELDS.issubset(obj.keys()):
            return _normalize_contract(obj)
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _normalize_contract(obj: dict) -> Optional[dict]:
    """Normalize aliases and reject contracts that cannot guide validation."""
    validate_aliases = {"1:1": "one_to_one", "1:m": "one_to_many",
                        "m:1": "many_to_one", "m:m": "many_to_many"}
    obj = dict(obj)
    def normalize_key(value):
        if isinstance(value, list):
            return [tuple(item) if isinstance(item, list) else item for item in value]
        return value

    obj["validate"] = validate_aliases.get(obj.get("validate"), obj.get("validate"))
    obj.setdefault("left_on", obj.get("on_key"))
    obj.setdefault("right_on", obj.get("on_key"))
    obj["on_key"] = normalize_key(obj.get("on_key"))
    obj["left_on"] = normalize_key(obj.get("left_on"))
    obj["right_on"] = normalize_key(obj.get("right_on"))
    obj.setdefault("suffixes_required", False)
    if obj.get("how") not in {"left", "inner", "outer", "right", "unknown"}:
        return None
    if obj.get("validate") not in {None, "one_to_one", "one_to_many", "many_to_one", "many_to_many"}:
        return None
    if obj.get("confidence") not in {"high", "medium", "low"}:
        obj["confidence"] = "low"
    return obj


def _fix_python_to_json(text: str) -> str:
    """Convert Python dict-like syntax to valid JSON."""
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)
    text = text.replace("'", '"')
    return text


def _extract_json_from_response(response: str) -> Optional[dict]:
    """
    Multi-layer JSON extraction from SLM-2 response.
    """
    if not response or not response.strip():
        return None

    cleaned = response.strip()
    cleaned = re.sub(r'```(?:json|python)?\s*', '', cleaned)
    cleaned = re.sub(r'```', '', cleaned)
    cleaned = cleaned.strip()

    result = _try_parse_json(cleaned)
    if result:
        return result

    # Bracket-counting extraction
    json_candidates = []
    i = 0
    while i < len(cleaned):
        if cleaned[i] == '{':
            depth = 0
            start = i
            for j in range(i, len(cleaned)):
                if cleaned[j] == '{':
                    depth += 1
                elif cleaned[j] == '}':
                    depth -= 1
                    if depth == 0:
                        json_candidates.append(cleaned[start:j+1])
                        break
            i = j + 1 if depth == 0 else i + 1
        else:
            i += 1

    json_candidates.sort(key=len, reverse=True)
    for candidate in json_candidates:
        result = _try_parse_json(candidate)
        if result:
            return result

    for candidate in json_candidates:
        fixed = _fix_python_to_json(candidate)
        result = _try_parse_json(fixed)
        if result:
            return result

    return None


# =====================================================================
#  Public API
# =====================================================================

def infer_contract(original_code: str, refactored_code: str) -> Optional[dict]:
    """
    Infer the behavioral contract for a merge refactoring.

    The candidate is not used to define its own oracle.

    Returns a dict with contract fields, or None on complete failure.
    """
    print(f"  [DEBUG:contract] infer_contract called")
    print(f"  [DEBUG:contract] original_code length: {len(original_code.strip())}")
    evidence = infer_contract_from_ast(original_code)

    # Tier 1: SLM-2. Only the original is supplied to prevent oracle leakage.
    try:
        print(f"  [DEBUG:contract] Trying SLM-2 inference...")
        query_model = _get_query_model()
        prompt = CONTRACT_PROMPT_TEMPLATE.format(
            original_code=original_code,
        )
        response = query_model(prompt, max_new_tokens=512, component="SLM-2")
        print(f"  [DEBUG:contract] SLM-2 response length: {len(response) if response else 0}")
        print(f"  [DEBUG:contract] SLM-2 response preview: {response[:300] if response else 'None'}...")

        slm_contract = _extract_json_from_response(response)
        if slm_contract:
            print(f"  [DEBUG:contract] ✓ SLM-2 JSON extraction succeeded")
            contract = _normalize_contract(slm_contract)
            if contract and evidence:
                # Literal schema/cardinality evidence is more reliable than
                # generated column names. SLM-2 remains responsible for intent.
                for field in ("on_key", "left_on", "right_on", "validate", "suffixes_required"):
                    contract[field] = evidence.get(field)
                if contract.get("how") == "unknown":
                    contract["how"] = evidence["how"]
                if evidence.get("how_source") in {"explicit", "context"}:
                    contract["how"] = evidence["how"]
                how, validate = contract.get("how"), contract.get("validate")
                contract["left_rows_preserved"] = how in {"left", "outer"}
                contract["right_rows_preserved"] = how in {"right", "outer"}
                if how == "left" and validate in {"one_to_one", "many_to_one"}:
                    contract["row_count_invariant"] = "eq_left"
                elif how == "outer":
                    contract["row_count_invariant"] = "gte_max"
                elif how == "inner" and validate == "one_to_one":
                    contract["row_count_invariant"] = "lte_min"
                else:
                    contract["row_count_invariant"] = "unknown"
            return contract
        else:
            print(f"  [DEBUG:contract] ✗ SLM-2 JSON extraction failed")
    except Exception as e:
        print(f"  [DEBUG:contract] ✗ SLM-2 call failed: {e}")

    # Tier 2: deterministic, candidate-independent fallback.
    if evidence:
        print("  [DEBUG:contract] Using deterministic original-code fallback")
        return evidence
    print(f"  [DEBUG:contract] ✗ All inference methods failed")
    return None
