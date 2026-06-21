import json
import re

text = """```

## Solution
```python
import pandas as pd

# Original Code
def merge_with_dups():
    left = pd.DataFrame({'key': [1,2], 'val': ['a','b']})
    right = pd.DataFrame({'key': [1,1,2], 'val2': ['x','y','z']})
    result = left.merge(right, on='key')
    return result

# Refactored Code
def merge_without_dups():
    left = pd.DataFrame({'key': [1,2], 'val': ['a','b']})
    right = pd.DataFrame({'key': [1,1,2], 'val2': ['x','y','z']})
    result = left.merge(right, on='key', how='left') # Using ...
"""

from models.slm2_merge_contract import _extract_json_from_response

res = _extract_json_from_response(text)
print("Res:", res)
