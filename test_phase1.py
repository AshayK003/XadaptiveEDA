import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from recommendation_engine import RecommendationEngine
from preference_learner import PreferenceTracker
from insight_generator import explain_recommendation, compare_recommendations
from data_quality import QualityReport, cleanse
from data_processor import DataProcessor
from visualization_generator import VisualizationGenerator
import pandas as pd
import numpy as np

profile = {
    'shape': (100, 5),
    'dtypes': {'x': 'float64', 'y': 'int64', 'cat': 'object', 'date': 'object', 'sparse': 'float64'},
    'numerical_cols': ['x', 'y', 'sparse'],
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

# ─── Test 1: Diversity penalty ───
print("=== Test 1: Diversity Penalty ===")
recs = engine.generate_recommendations(profile, prefs, quality_score=0.72)
has_diversity = any(r.get('diversity_penalty') is not None for r in recs)
print(f"  Some recs have diversity_penalty: {has_diversity}")
# Distribution and outliers both operate on 'numerical' — at least one should be penalized
num_types = [r['type'] for r in recs]
dist_idx = next((i for i, t in enumerate(num_types) if t == 'distribution'), -1)
out_idx = next((i for i, t in enumerate(num_types) if t == 'outliers'), -1)
if dist_idx >= 0 and out_idx >= 0:
    assert recs[max(dist_idx, out_idx)].get('diversity_penalty') == 0.85, f"Second numerical rec should be penalized"
    print("  Numerical duplication correctly penalized.")
print("  Passed!")

# ─── Test 2: Implicit tracking ───
print("\n=== Test 2: Implicit Tracking ===")
tracker = PreferenceTracker()
rec = {'type': 'distribution', 'title': 'Distribution Analysis', 'score': 0.5}
r1 = tracker.track_interaction(rec, 'explored', metadata={"event": "show_explanation"})
assert len(tracker.interaction_history) == 1
assert tracker.interaction_history[0]['action'] == 'explored'
assert tracker.interaction_history[0].get('metadata', {}).get('event') == 'show_explanation'
assert abs(tracker.preference_weights['distribution'] - 0.55) < 0.001, f"Expected ~0.55, got {tracker.preference_weights['distribution']}"

r2 = tracker.track_interaction(rec, 'column_selected', metadata={"columns": ["x"]})
assert abs(tracker.preference_weights['distribution'] - 0.58) < 0.001, f"Expected ~0.58, got {tracker.preference_weights['distribution']}"
print("  Explored: 0.05 increment correct")
print("  Column selected: 0.03 increment correct")
print("  Passed!")

# ─── Test 3: Comparative explanation ───
print("\n=== Test 3: Comparative Explanation ===")
recs = engine.generate_recommendations(profile, prefs)
comparison = compare_recommendations(recs[0], recs[1])
assert 'Base score' in comparison
assert 'Data relevance' in comparison
assert 'Your priority' in comparison
assert 'Final score' in comparison
assert recs[0]['title'] in comparison
assert recs[1]['title'] in comparison
print(f"  Comparison contains: Base score, Data relevance, Priority, Final score")
print("  Passed!")

# ─── Test 4: Existing functionality not broken ───
print("\n=== Test 4: Regression Checks ===")

# 4a: DataProcessor.cleanse still works
dp = DataProcessor()
import tempfile, os
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    f.write('NA,name\n1,alice\nN/A,bob\n')
    tmp = f.name
df = dp.load_data(tmp)
clean, report = dp.cleanse(df)
assert clean.shape[0] > 0
os.unlink(tmp)
print("  4a. DataProcessor.cleanse: OK")

# 4b: Visualization still works
vg = VisualizationGenerator()
fig = vg.generate_visualization('distribution', clean, ['na'], quality_report=report)
assert fig is not None
assert len(fig.data) > 0
print("  4b. Visualization: OK")

# 4c: Explain still works
rec = {'type': 'distribution', 'title': 'Distribution Analysis', 'score': 0.5, 'base_score': 0.8,
       'pref_score': 0.5, 'data_relevance': 0.7, 'quality_adjustment': 0.86,
       'techniques': ['histogram'], 'data_factors': ['2 numerical cols'], 'columns': ['x']}
exp = explain_recommendation(rec, profile, prefs, qr)
assert len(exp['reasons']) > 0
assert len(exp['technique_reasons']) > 0
print("  4c. explain_recommendation: OK")

# 4d: Preferences validation still works
tracker2 = PreferenceTracker()
try:
    tracker2.set_preferences({'distribution': 0.5})
    assert False, "Should have raised ValueError for missing keys"
except ValueError:
    pass
tracker2.set_preferences(prefs)
assert tracker2.preference_weights == prefs
print("  4d. set_preferences validation: OK")

print("\n=== ALL PHASE 1 TESTS PASSED ===")
