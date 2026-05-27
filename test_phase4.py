import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from nlq_engine import match_query, format_result

# ─── Test 1: Basic keyword matches ───
print("=== Test 1: Keyword Matches ===")
tests = [
    ("show me outliers in the data", 'outliers'),
    ("what columns have missing values", 'missing_values'),
    ("correlation between variables", 'correlation'),
    ("distribution of ages", 'distribution'),
    ("categorical columns summary", 'categorical'),
    ("time series trend analysis", 'time_series'),
    ("anomaly detection", 'outliers'),
    ("histogram of salary", 'distribution'),
    ("null values", 'missing_values'),
    ("bar chart of categories", 'categorical'),
]
for query, expected in tests:
    result, confidence = match_query(query)
    assert result == expected, f"'{query}' -> {result}, expected {expected}"
    assert confidence > 0
print(f"  All {len(tests)} queries correctly matched. Passed!")

# ─── Test 2: No match ───
print("\n=== Test 2: No Match ===")
result, confidence = match_query("hello world this is a test")
assert result is None
assert confidence == 0
print("  Non-matching query returns None. Passed!")

# ─── Test 3: Empty query ───
print("\n=== Test 3: Empty Query ===")
result, confidence = match_query("")
assert result is None
print("  Empty query returns None. Passed!")

# ─── Test 4: Case insensitive ───
print("\n=== Test 4: Case Insensitivity ===")
result, confidence = match_query("OUTLIERS IN DATA")
assert result == 'outliers'
print("  Uppercase query matches. Passed!")

# ─── Test 5: format_result ───
print("\n=== Test 5: Format Result ===")
fr = format_result('outliers', 0.7)
assert fr is not None
assert fr['type'] == 'outliers'
assert fr['title'] == 'Outlier Detection'
assert fr['confidence'] == 0.7
print("  Format result returns correct dict. Passed!")

# ─── Test 6: format_result None ───
print("\n=== Test 6: Format Result None ===")
fr = format_result(None, 0)
assert fr is None
print("  None input returns None. Passed!")

print("\n=== ALL PHASE 4 TESTS PASSED ===")
