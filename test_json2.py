import json

obj = json.loads("""{
    "how": null,
    "row_count_invariant": null,
    "left_rows_preserved": null,
    "right_rows_preserved": null,
    "on_key": null,
    "validate": null,
    "confidence": "low",
    "reasoning": "The refactored code does not explicitly specify how parameter so we assume default value which is inner."
}""")
print(obj)
