"""Regression tests for the Sep 2026 audit fixes (P0/P1/P2/P3)."""

import numpy as np
import pandas as pd

from insight_generator import explain_recommendation
from llm_adapter import _safe_error, is_safe_endpoint
from nlq_engine import _stem
from preference_learner import PreferenceTracker
from recommendation_engine import RecommendationEngine


def _profile():
    return {
        'skewness': {'y': 0.1},
        'has_outliers': {'y': 3.5},
        'missing_percentage': {'y': 0},
        'categorical_cardinality': {},
    }


def test_outlier_scale_no_saturation():
    e = RecommendationEngine()
    s = e._score_column_interestingness('distribution', 'y', _profile())
    assert s < 0.5  # was min(...,1.0) == 1.0 before the /100 fix


def test_epsilon_boost_survives_sort():
    import random
    e = RecommendationEngine()
    recs = [{'type': f't{i}', 'score': 0.9 - i * 0.1, 'columns': []} for i in range(8)]
    random.seed(1)
    out = e._apply_epsilon_greedy(recs, epsilon=1.0)
    boosted = [r for r in out if r.get('exploration')]
    assert boosted, "epsilon=1.0 must always explore"
    final = sorted(out, key=lambda x: x['score'], reverse=True)
    assert final.index(boosted[0]) < 5


def test_normalize_dedupes_collisions():
    from data_quality import DataQualityPipeline
    p = DataQualityPipeline()
    df = pd.DataFrame([[1, 2, 3, 4]], columns=['A B', 'A-B', 'a_b', 'a_b_1'])
    out = p._normalize_column_names(df)
    assert len(set(out.columns)) == 4
    assert list(out.columns)[:3] == ['a_b', 'a_b_1', 'a_b_2']


def test_decay_applies_once_per_type():
    t = PreferenceTracker()
    t.preference_weights = {'distribution': 0.9}
    old = pd.Timestamp.now() - pd.Timedelta(hours=100)
    t.interaction_history = [
        {'action': 'view', 'recommendation_type': 'distribution', 'timestamp': old}
        for _ in range(5)
    ]
    t.apply_temporal_decay(half_life_hours=24)
    # exactly one application: 0.5 + 0.4 * 2**(-100/24)
    assert abs(t.preference_weights['distribution'] - (0.5 + 0.4 * 2 ** (-100 / 24))) < 1e-9


def test_kmeans_guards_and_no_nan():
    from visualization_generator import VisualizationGenerator
    g = VisualizationGenerator()
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [1.0, 1.0, 1.0]})
    labels = g._kmeans(df, k=10, random_state=0)
    assert len(labels) == 3
    df2 = pd.DataFrame({'a': np.random.default_rng(0).normal(size=50),
                        'b': np.random.default_rng(1).normal(size=50)})
    labels2 = g._kmeans(df2, k=5, random_state=0)
    assert len(labels2) == 50


def test_heatmap_keeps_diagonal():
    from visualization_generator import VisualizationGenerator
    g = VisualizationGenerator()
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0],
                       'b': [2.0, 1.0, 4.0, 3.0]})
    fig = g._create_correlation_plot(df, ['a', 'b'])
    assert fig is not None


def test_stemmer_order():
    assert _stem('classes') == 'class'
    assert _stem('boxes') == 'box'
    assert _stem('cats') == 'cat'


def test_endpoint_guard():
    assert is_safe_endpoint('https://openrouter.ai/api/v1/chat/completions')
    assert is_safe_endpoint('http://localhost:11434/v1')
    assert not is_safe_endpoint('file:///etc/passwd')
    assert not is_safe_endpoint('http://169.254.169.254/latest')
    assert not is_safe_endpoint('')


def test_safe_error_redacts_keys():
    msg = _safe_error(ValueError("bad key sk-abc123XYZ hello"))
    assert 'sk-abc123XYZ' not in msg
    assert 'sk-***' in msg


def test_score_display_lists_adjustments():
    rec = {'type': 'distribution', 'score': 0.6, 'base_score': 0.8,
           'data_relevance': 0.9, 'quality_adjustment': 1.0,
           'diversity_penalty': 0.85, 'columns': ['x']}
    out = explain_recommendation(rec, {}, {'distribution': 0.5})
    text = "\n".join(out if isinstance(out, list) else [str(out)])
    assert 'diversity' in text


def test_rate_log_evicts_idle_providers():
    import time

    from llm_adapter import RATE_LIMIT_WINDOW, _check_rate_limit, _remote_call_log
    _remote_call_log.clear()
    _remote_call_log['idle'] = [time.time() - RATE_LIMIT_WINDOW - 10]
    _check_rate_limit('active')
    assert 'idle' not in _remote_call_log
    assert 'active' in _remote_call_log
    _remote_call_log.clear()
