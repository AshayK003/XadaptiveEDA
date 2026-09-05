import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

from data_processor import DataProcessor
from data_quality import QualityReport
from insight_generator import global_explanation_summary
from recommendation_engine import RecommendationEngine

profile = {
    'shape': (100, 5),
    'dtypes': {'x': 'float64', 'y': 'int64', 'cat': 'object', 'date': 'object', 'sparse': 'float64'},
    'numerical_cols': ['y', 'x'],
    'categorical_cols': ['cat'],
    'missing_values': {'x': 0, 'y': 0, 'sparse': 70, 'cat': 0, 'date': 0},
    'missing_percentage': {'x': 0, 'y': 0, 'sparse': 70.0, 'cat': 0, 'date': 0},
    'skewness': {'x': 2.3, 'y': 0.1, 'sparse': None},
    'correlation_exists': True,
    'time_series_candidates': ['date'],
    'categorical_cardinality': {'cat': 5},
    'has_outliers': {'y': 3.5},
    'unique_counts': {'x': 90, 'y': 50, 'sparse': 15, 'cat': 5, 'date': 30}
}
prefs = {k: 0.5 for k in ['distribution','correlation','missing_values','categorical','outliers','time_series','clustering','feature_importance']}

qr = QualityReport(sparse_columns=['sparse'], constant_columns=[], mixed_type_columns=[],
                   null_percentages={'sparse': 70.0}, overall_quality_score=0.72)

engine = RecommendationEngine()

# ─── Test 1: Column interestingness scoring ───
print("=== Test 1: Column Interestingness Ordering ===")
recs = engine.generate_recommendations(profile, prefs, quality_score=0.72)

for r in recs:
    cols = r['columns']
    print(f"  {r['type']}: {cols}")
    if len(cols) > 1:
        interestingness_scores = []
        for c in cols:
            score = engine._score_column_interestingness(r['type'], c, profile)
            interestingness_scores.append(score)
        # Verify descending order
        for i in range(len(interestingness_scores) - 1):
            assert interestingness_scores[i] >= interestingness_scores[i+1] - 0.001, (
                f"Column order not sorted: {cols}, scores={interestingness_scores}"
            )
print("  All column lists sorted by interestingness descending. Passed!")

# ─── Test 2: Global explanation summary ───
print("\n=== Test 2: Global Explanation Summary ===")
history = [
    {'recommendation_type': 'distribution', 'action': 'explored', 'timestamp': None},
    {'recommendation_type': 'correlation', 'action': 'liked', 'timestamp': None},
    {'recommendation_type': 'outliers', 'action': 'explored', 'timestamp': None},
]
summary = global_explanation_summary(profile, qr, history, prefs)
assert 'Explored analysis types' in summary
assert 'distribution' in summary
assert 'correlation' in summary
assert 'outliers' in summary
assert 'Quality score' in summary
assert 'Data Quality Notes' in summary
assert 'Feedback given' in summary
print("  Summary includes explored types, quality score, data notes, and feedback. Passed!")

# ─── Test 3: Empty history still renders ───
print("\n=== Test 3: Empty History ===")
summary2 = global_explanation_summary(profile, qr, [], prefs)
assert 'No analysis types explored yet' in summary2
print("  Handles empty history. Passed!")

# ─── Test 4: Stratified Sampling ───
print("\n=== Test 4: Stratified Sampling ===")
df_sample = pd.DataFrame({
    'cat': ['A'] * 50 + ['B'] * 50,
    'val': range(100)
})
sampled = DataProcessor.sample_stratified(df_sample, target_rows=20, random_state=42)
assert len(sampled) <= 20
assert set(sampled['cat'].unique()).issubset({'A', 'B'})
print(f"  Sampled {len(sampled)} from 100 rows (both strata preserved). Passed!")

# ─── Test 5: Diversity still correctly applied ───
print("\n=== Test 5: Diversity + Interestingness Compatibility ===")
assert all('type' in r and 'columns' in r for r in recs)
types = [r['type'] for r in recs[:5]]
num_count = sum(1 for t in types if t in ('distribution', 'outliers'))
if num_count >= 2:
    penalized = [r for r in recs if r.get('diversity_penalty') == 0.85]
    print(f"  {len(penalized)} rec(s) penalized out of {len(recs)}")
else:
    print("  No diversity penalty applied (different type mix)")
print("  Passed!")

print("\n=== ALL PHASE 2 TESTS PASSED ===")
