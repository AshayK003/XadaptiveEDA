import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from data_quality import cleanse, QualityReport, DataQualityPipeline

# ─── Test 1: Empty DataFrame ───
print("=== Test 1: Empty DataFrame ===")
df_empty = pd.DataFrame()
clean, report = cleanse(df_empty)
assert clean.empty
assert len(report.warnings) > 0
print("  Empty DF returns empty + warning. Passed!")

# ─── Test 2: Missing token normalization ───
print("\n=== Test 2: Missing Token Normalization ===")
df = pd.DataFrame({'a': ['NA', 'N/A', 'NULL', '', '-', '?', '#N/A', 'hello', 'world'], 'b': [1]*9})
clean, report = cleanse(df)
# 7 tokens normalized to NaN in col 'a', rows with all-NaN in ALL cols are removed
# col 'b' has value 1 in all rows, so no rows removed
nulls_in_a = clean['a'].isna().sum()
assert nulls_in_a == 7, f"Expected 7 nulls, got {nulls_in_a}"
assert clean['a'].iloc[-2] == 'hello' or clean['a'].iloc[-1] == 'world'
print(f"  {nulls_in_a}/9 tokens normalized to NaN. Passed!")

# ─── Test 3: Duplicate columns ───
print("\n=== Test 3: Duplicate Column Dedup ===")
# Create DataFrame programmatically to avoid pandas column dedup
data = [[1, 3, 5], [2, 4, 6]]
df = pd.DataFrame(data)
df.columns = ['x', 'x', 'y']  # Force duplicate column names
clean, report = cleanse(df)
assert clean.columns[0] == 'x'
assert clean.columns[1] != 'x' or clean.columns[1] == 'x_1'
print(f"  Columns: {list(clean.columns)}. Passed!")

# ─── Test 4: Column name normalization ───
print("\n=== Test 4: Column Name Normalization ===")
df = pd.DataFrame({'My Column': [1], 'another-COLUMN!': [2], '  spaced  ': [3]})
clean, report = cleanse(df)
assert 'my_column' in clean.columns
assert 'another_column' in clean.columns
assert 'spaced' in clean.columns
print(f"  Columns: {list(clean.columns)}. Passed!")

# ─── Test 5: Infinite values ───
print("\n=== Test 5: Infinite Value Replacement ===")
df = pd.DataFrame({'a': [1.0, np.inf, -np.inf, np.nan, 5.0], 'b': [1]*5})
clean, report = cleanse(df)
nulls_in_a = clean['a'].isna().sum()
assert nulls_in_a == 3, f"Expected 3 NaNs (inf, -inf, nan), got {nulls_in_a}"
assert clean['a'].iloc[0] == 1.0
print(f"  {nulls_in_a} inf/-inf/nan replaced. Passed!")

# ─── Test 6: Empty column/row removal ───
print("\n=== Test 6: Empty Column & Row Removal ===")
df = pd.DataFrame({'a': [1, 2, np.nan], 'b': [np.nan, np.nan, np.nan], 'c': [np.nan, np.nan, 3]})
clean, report = cleanse(df)
assert 'b' not in clean.columns, "All-null column should be removed"
assert 'a' in clean.columns
assert 'c' in clean.columns
print(f"  Columns after: {list(clean.columns)}. Passed!")

# ─── Test 7: Sparse column detection ───
print("\n=== Test 7: Sparse Column Detection ===")
df = pd.DataFrame({'a': [1, 2, 3], 'b': [np.nan, np.nan, 1], 'c': [np.nan, np.nan, np.nan], 'keep': [1]*3})
# 'c' will be removed as empty, 'b' has 2/3 null > 50% → sparse
clean, report = cleanse(df)
assert len(report.sparse_columns) > 0, f"No sparse columns detected: {report.sparse_columns}"
assert 'b' in report.sparse_columns
print(f"  Sparse columns: {report.sparse_columns}. Passed!")

# ─── Test 8: Constant column detection ───
print("\n=== Test 8: Constant Column Detection ===")
df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'x', 'x'], 'c': [np.nan, np.nan, np.nan], 'keep': [1]*3})
clean, report = cleanse(df)
assert len(report.constant_columns) > 0, f"No constant columns detected: {report.constant_columns}"
print(f"  Constant columns: {report.constant_columns}. Passed!")

# ─── Test 9: type inference ───
print("\n=== Test 9: Type Inference ===")
df = pd.DataFrame({'a': ['1', '2', '3.5'], 'b': ['x', 'y', 'z'], 'c': ['2021-01-01', '2021-01-02', '2021-01-03'], 'keep': [1]*3})
clean, report = cleanse(df)
assert pd.api.types.is_numeric_dtype(clean['a']), "a should be numeric"
assert pd.api.types.is_string_dtype(clean['b']), "b should stay string"
print(f"  Types: a={clean['a'].dtype}, b={clean['b'].dtype}. Passed!")

# ─── Test 10: Quality report structure ───
print("\n=== Test 10: Quality Report Structure ===")
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
clean, report = cleanse(df)
assert hasattr(report, 'overall_quality_score')
assert hasattr(report, 'completeness')
assert hasattr(report, 'uniqueness')
assert hasattr(report, 'memory_usage_mb')
assert report.overall_quality_score > 0
assert report.overall_quality_score <= 1.0
print(f"  Quality score: {report.overall_quality_score:.2f}. Passed!")

# ─── Test 11: Mixed type detection ───
print("\n=== Test 11: Mixed Type Detection ===")
df = pd.DataFrame({'a': [1, 'hello', 3.5, None], 'b': ['x', 'y', 'z', 'w'], 'keep': [1]*4})
clean, report = cleanse(df)
assert len(report.mixed_type_columns) > 0, f"No mixed types detected: {report.mixed_type_columns}"
print(f"  Mixed-type columns: {report.mixed_type_columns}. Passed!")

# ─── Test 12: Nondestructive (original df unchanged) ───
print("\n=== Test 12: Non-destructive ===")
original = pd.DataFrame({' A ': [1, np.inf], 'keep': [1, 2]})
clean, report = cleanse(original)
# _normalize_column_names changes ' A ' → 'a', so check the original
assert ' A ' in original.columns
assert original[' A '].iloc[1] == np.inf
print("  Original DataFrame unchanged. Passed!")

# ─── Test 13: Warnings reset between run() calls (Issue #1) ───
print("\n=== Test 13: Warnings Reset Between Runs ===")
pipeline = DataQualityPipeline()
df_warn = pd.DataFrame({'a': [1, 1, 1]})     # constant column → triggers warnings
df_clean = pd.DataFrame({'b': [1, 2, 3]})    # clean data → no warnings expected
_, report1 = pipeline.run(df_warn)
assert len(report1.warnings) > 0, "First run should produce warnings"
_, report2 = pipeline.run(df_clean)
assert len(report2.warnings) == 0, f"Second run leaked warnings: {report2.warnings}"
assert report1.warnings is not report2.warnings, "Reports should not share the same warnings list"
assert len(report1.warnings) > 0, "First report's warnings should be unchanged after second run"
print("  Warnings reset between runs, reports independent. Passed!")

print("\n=== ALL DATA QUALITY PIPELINE TESTS PASSED ===")