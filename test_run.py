# test_run.py
from models.slm_refactor import refactor_gnc

# The code string you want to test
code_snippet = """
for epoch in range(10):
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
"""

# Run the refactoring function
result = refactor_gnc(code_snippet)

# Print the results cleanly
print("--- TEST RESULTS ---")
print('Success:', result.success)
print('Reasoning:', result.reasoning)
print('\nRefactored code:')
print(result.refactored_code)
if result.error:
    print('Error Details:', result.error)