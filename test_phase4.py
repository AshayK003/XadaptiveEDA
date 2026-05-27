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
    result, confidence, cols = match_query(query)
    assert result == expected, f"'{query}' -> {result}, expected {expected}"
    assert confidence > 0
print(f"  All {len(tests)} queries correctly matched. Passed!")

# ─── Test 2: No match ───
print("\n=== Test 2: No Match ===")
result, confidence, cols = match_query("hello world this is a test")
assert result is None
assert confidence == 0
assert cols == []
print("  Non-matching query returns (None, 0, []). Passed!")

# ─── Test 3: Empty query ───
print("\n=== Test 3: Empty Query ===")
result, confidence, cols = match_query("")
assert result is None
print("  Empty query returns None. Passed!")

# ─── Test 4: Case insensitive ───
print("\n=== Test 4: Case Insensitivity ===")
result, confidence, cols = match_query("OUTLIERS IN DATA")
assert result == 'outliers'
print("  Uppercase query matches. Passed!")

# ─── Test 5: format_result ───
print("\n=== Test 5: Format Result ===")
fr = format_result('outliers', 0.7)
assert fr is not None
assert fr['type'] == 'outliers'
assert fr['title'] == 'Outlier Detection'
assert fr['confidence'] == 0.7
assert fr['columns'] == []
print("  Format result returns correct dict with columns. Passed!")

# ─── Test 6: format_result None ───
print("\n=== Test 6: Format Result None ===")
fr = format_result(None, 0)
assert fr is None
print("  None input returns None. Passed!")

# ─── Test 7: Column name extraction ───
print("\n=== Test 7: Column Name Extraction ===")
cols = ['duration', 'score', 'age', 'income', 'category']
result, conf, mentioned = match_query("correlation between duration and score", columns=cols)
assert result == 'correlation', f"Expected correlation, got {result}"
assert conf > 0.7
assert 'duration' in mentioned, f"Expected 'duration' in {mentioned}"
assert 'score' in mentioned, f"Expected 'score' in {mentioned}"
print(f"  Matched '{result}' (confidence {conf}), extracted columns: {mentioned}. Passed!")

# ─── Test 8: Column names in query boost confidence ───
print("\n=== Test 8: Column Name Boost ===")
result_no_col, conf_no_col, _ = match_query("correlation between variables")
result_with_col, conf_with_col, mentioned = match_query("correlation between duration and score", columns=cols)
assert conf_with_col > conf_no_col, "Column mention should boost confidence"
assert len(mentioned) == 2
print(f"  Confidence: no-cols={conf_no_col}, with-cols={conf_with_col}, extracted={mentioned}. Passed!")

# ─── Test 9: Stop word filtering ───
print("\n=== Test 9: Stop Word Filtering ===")
result, confidence, cols = match_query("please show me the distribution analysis")
assert result == 'distribution'
print(f"  Stop words filtered, matched: {result}. Passed!")

# ─── Test 10: Synonym matching ───
print("\n=== Test 10: Synonym Matching ===")
result, confidence, cols = match_query("find anomalies in the data")
assert result == 'outliers', f"Expected outliers (via anomaly->outlier synonym), got {result}"
print(f"  Synonym match: 'find anomalies' -> {result}. Passed!")

# ─── Test 11: Format result with columns ───
print("\n=== Test 11: Format Result with Columns ===")
fr = format_result('correlation', 0.95, mentioned_columns=['duration', 'score'])
assert fr['columns'] == ['duration', 'score']
assert fr['type'] == 'correlation'
print(f"  Format result columns: {fr['columns']}. Passed!")

# ─── Test 12: Column extraction with partial match ───
print("\n=== Test 12: Partial Column Match ===")
result, confidence, mentioned = match_query("show outliers in revenue_cost", columns=['revenue', 'cost', 'profit'])
assert result == 'outliers'
assert 'revenue_cost' not in mentioned
print(f"  Non-matching column name excluded: {mentioned}. Passed!")

print("\n=== ALL PHASE 4 TESTS PASSED ===")
