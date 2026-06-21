from models.slm2_merge_contract import _extract_json_from_response

text = """## Solution
Here is the solution for the problem:

```python
import json

# The behavioral contract fields
contract = {
    "how": None,
    "row_count_invariant": None,
    "left_rows_preserved": None,
    "right_rows_preserved": None,
    "on_key": None,
    "validate": None,
    "confidence": "low",
    "reasoning": "The refactored code does not explicitly specify how to perform the merge operation."
}
```
"""
result = _extract_json_from_response(text)
print("Result:", result)
