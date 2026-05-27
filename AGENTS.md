# Project Memory

## Identity
EDA Assistant — Streamlit app that profiles datasets, ranks 6 analysis types by data relevance × user priority, and renders plotly visualizations.

## Architecture
```
app.py (Streamlet UI, 376 lines)
  ├── data_processor.py      (162 lines) — load_data(), profile_dataset()
  ├── recommendation_engine.py (232 lines) — scores/ranks 6 analysis types
  ├── preference_learner.py   (80 lines)  — PreferenceTracker class
  ├── insight_generator.py    (135 lines) — generate_insights(), explain_recommendation(), explain_user_preferences()
  ├── visualization_generator.py (174 lines) — VisualizationGenerator class
  └── constants.py            (17 lines)  — ANALYSIS_TYPES, DEFAULT_PREFERENCES
```

## Patterns & Conventions

### Session State Keys
Always check/initialize in `app.py` lines 41-54:
- `user_preferences` — dict[str, float], cloned from DEFAULT_PREFERENCES
- `interaction_history` — list of dicts
- `data_profile` — dict from DataProcessor.profile_dataset()
- `recommendations` — list of dicts from RecommendationEngine
- `df` — the DataFrame itself
- `last_file` — previous filename for cache invalidation
- Badge toggle keys: `show_why_{type}_{i}` — must include index for position independence

### Scoring Formula
`final_score = base_score × user_preference_score × data_relevance`
- base_score: 0.6–0.9 (catalog in recommendation_engine.py:16-58)
- user_preference_score: 0.1–1.0 (optional slider or feedback-adjusted)
- data_relevance: 0.5–1.0 (computed from actual data characteristics in _calculate_data_relevance)
- Top 5 results shown, sorted descending

### Recommendation Dict Shape
```python
{
    'type': str,           # one of ANALYSIS_TYPES
    'title': str,          # e.g. "Distribution Analysis"
    'description': str,    # from catalog
    'techniques': list[str],  # e.g. ['histogram', 'kde']
    'score': float,        # 0–1
    'data_factors': list,  # human-readable factor labels
    'columns': list[str]   # applicable column names
}
```

### PreferenceTracker (preference_learner.py)
- Fixed delta adjustments, NOT ML:
  - liked: +0.10, selected: +0.05, disliked: −0.10, ignored: −0.02
  - Clamped to [0.1, 1.0]
- set_preferences() validates all keys present and values in range; logs a manual_adjustment interaction
- get_interaction_stats() returns totals only — no UI currently uses this

### Data Profile Schema (data_processor.py:32-48)
```python
{
    'shape': (int, int),
    'dtypes': dict[str, str],
    'missing_values': dict[str, int],
    'missing_percentage': dict[str, float],
    'numerical_cols': list[str],
    'categorical_cols': list[str],
    'unique_counts': dict[str, int],
    'skewness': dict[str, float|None],
    'correlation_exists': bool,
    'time_series_candidates': list[str],
    'categorical_cardinality': dict[str, int],
    'has_outliers': dict[str, float]  # column → % of rows beyond 1.5×IQR
}
```

### Visualization Dispatch (visualization_generator.py:9-17)
```python
dispatch = {
    'distribution': _create_distribution_plot,    # Histogram + Boxplot (subplots)
    'correlation': _create_correlation_plot,      # Triangular + Full heatmap (pearson only)
    'missing_values': _create_missing_values_plot, # Bar chart of null %
    'categorical': _create_categorical_plot,      # Bar or Pie (selectable)
    'outliers': _create_outliers_plot,            # Boxplot per column
    'time_series': _create_time_series_plot,      # Line chart against first numerical col
}
```
All return plotly `go.Figure`. Matplotlib/seaborn removed — do NOT import them.

### Insight Generator (insight_generator.py)
- `generate_insights(data_profile)` → list[str] — data-driven, no lookup tables
- `explain_recommendation(rec, data_profile, user_prefs)` → dict with 'reasons' and 'technique_reasons'
- `explain_user_preferences(preferences)` → plain-text summary

## Key Decisions (DO NOT REVERSE)

1. **No ML/AI** — PreferenceTracker uses fixed deltas, not learned models. UI must not claim "learning" or "AI."
2. **Session-only state** — preferences reset on page reload. No persistence layer.
3. **Plotly only** — matplotlib/seaborn/scipy removed from both code and requirements. All charts interactive.
4. **Pure pandas** — Polars not used. Keep it consistent.
5. **Empty dataset guard** — profile_dataset() returns empty schema dict when df.empty; load_data raises ValueError on unsupported format.
6. **No format inference fallback** — removed because it masked real errors with misleading openpyxl messages.

## Critical Gotchas

- **Slider form pattern** — app.py uses `st.form()` for preferences with `st.form_submit_button()`. `set_preferences()` validates ALL required keys; missing any raises ValueError.
- **Toggle key uniqueness** — `'show_why_' + rec['type'] + '_' + str(i)` — the `str(i)` index is critical because two recommendations of the same type would otherwise share the same key.
- **Time series limitation** — plots against `df.select_dtypes(include=['number']).columns[0]` (first numerical col) only.
- **Correlation default** — selects first 5 numerical columns by default using `[:min(5, len(column_options))]`.
- **Categorical default** — selects first 3 low-cardinality columns (sorted by nunique ascending).
- **Interaction tracking** — column selection calls `preference_learner.track_interaction(rec, 'selected')` which permanently adjusts the weight. This means every multiselect change adds +0.05. Not debounced.
- **Outlier detection** — uses IQR (Q1-1.5×IQR, Q3+1.5×IQR). Reports percentage of rows flagged; does NOT return indices.
- **Second project copy** — `C:\Users\Ashay\X-adapativeEDA\` is maintained in sync with `D:\Personal projects\X-adapativeEDA\`. Changes should be copied unless user says otherwise.

## File Loading (data_processor.py)
- Supports: CSV, XLSX, XLS, JSON
- Streamlit `file_uploader` accepts these four extensions
- Raises `ValueError` on unsupported format (no fallback chain)

## Dependencies
```
streamlit>=1.36.0   (needs st.rerun(), st.toast())
pandas>=1.5.0
numpy>=1.23.0
plotly>=5.10.0
openpyxl>=3.1.0    (XLSX only)
xlrd>=2.0.0        (XLS only)
```

## Verification
- `streamlit run app.py` — serves on http://localhost:8501
- No test framework exists. Manual verification: upload `.xlsx` file, confirm profile loads, click through all 6 analysis types, verify 👍/👎 adjustments stick in sidebar chart.
- Lint: none configured. App is single-file pattern with no type annotations.

## Git
- Remote: `https://github.com/AshayK003/XadaptiveEDA.git` (branch: main)
- Init/initial commit message style: descriptive, lowercase prefix (e.g., "docs: rewrite README with module docs, data profile schema, analysis types, and limitations")
