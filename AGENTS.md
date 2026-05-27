# Project Memory

## Identity
EDA Assistant — Streamlit app that profiles datasets, ranks 6 analysis types by data relevance × user priority, and renders plotly visualizations.

## Architecture
```
├── app.py                    # Streamlit UI (orchestration, ~780 lines)
├── data_processor.py         # File loading, dataset cleansing, profiling
├── data_quality.py           # DataQualityPipeline, QualityReport, cleanse()
├── recommendation_engine.py  # Scoring/ranking, diversity, column interestingness
├── preference_learner.py     # PreferenceTracker: 5 actions + goal/decay/persistence
├── insight_generator.py      # Explanations, comparisons, global summary
├── visualization_generator.py# Plotly chart creation
├── constants.py              # ANALYSIS_TYPES, DEFAULT_PREFERENCES, ANALYSIS_GOALS
├── llm_adapter.py            # LLM analysis, NLQ classification, chat, column naming
├── nlq_engine.py             # NLP query classifier (stemming, synonyms, TF scoring)
├── requirements.txt
└── README.md
```

### Data Flow
```
Upload file → load_data() → raw df
                               ↓
                       cleanse() → (cleaned df, QualityReport)
                               ↓
                    profile_dataset() → profile dict
                               ↓
              User priorities → generate_recommendations() → ranked list
                               ↓
              User selects analysis → VisualizationGenerator → plotly figure
                               ↓
              User clicks 👍/👎 → PreferenceTracker → weights adjusted
```

## Patterns & Conventions

### Session State Keys
Always check/initialize in `app.py` lines 41-54:
- `user_preferences` — dict[str, float], cloned from DEFAULT_PREFERENCES
- `interaction_history` — list of dicts
- `data_profile` — dict from DataProcessor.profile_dataset()
- `recommendations` — list of dicts from RecommendationEngine
- `df` — the DataFrame itself (post-cleanse)
- `quality_report` — QualityReport dataclass from cleanse()
- `last_file` — previous filename for cache invalidation
- `sample_large_dataset` — bool checkbox for progressive sampling (only created when >50k rows)
- `ai_enabled` — bool toggle for AI insights
- `_active_goal` — string name of current analysis goal
- `_expert_mode` — bool for expert/beginner toggle
- `_nlq_match` — string from NLQ bar match result
- `_finalized` — bool, set after user clicks Finalize Dataset
- `_cols_renamed` — bool, set after column rename step
- `_rows_dropped` — int, number of rows dropped (set after drop step)
- `_cardinality` — dict[str, int], cached nunique values
- `_chat_enabled` — bool, separate toggle for the chat interface (independent of ai_enabled)
- `_chat_history` — list[dict], chat conversation history for "Ask anything" feature
- Badge toggle keys: `show_why_{type}_{i}` — must include index for position independence

### Scoring Formula
`final_score = base_score × user_preference_score × data_relevance × quality_adjustment [× diversity_penalty]`
- base_score: 0.6–0.9 (catalog in recommendation_engine.py:16-58)
- user_preference_score: 0.1–1.0 (optional slider or feedback-adjusted)
- data_relevance: 0.5–1.0 (computed from actual data characteristics in _calculate_data_relevance)
- quality_adjustment: 0.5 + 0.5 × quality_score (0.5–1.0, only if quality_score provided)
- diversity_penalty: 0.85 if same-category duplicate, else 1.0
- Top 5 results shown, sorted descending

### Recommendation Dict Shape
```python
{
    'type': str,           # one of ANALYSIS_TYPES
    'title': str,          # e.g. "Distribution Analysis"
    'description': str,    # from catalog
    'techniques': list[str],  # e.g. ['histogram', 'kde']
    'score': float,        # 0–1
    'base_score': float,   # 0.6–0.9 catalog value
    'pref_score': float,   # user preference weight
    'data_relevance': float, # 0.5–1.0
    'quality_adjustment': float|None,  # 0.5–1.0 or None
    'diversity_penalty': float|None,   # 0.85 or None
    'data_factors': list,  # human-readable factor labels
    'columns': list[str]   # applicable column names, sorted by interestingness
}
```

### PreferenceTracker (preference_learner.py)
- Fixed delta adjustments, NOT ML:
  - liked: +0.10, selected: +0.05, disliked: −0.10, ignored: −0.02
  - Column selection count: +0.05 per unique column selected
  - Clamped to [0.1, 1.0]
- set_preferences() validates all keys present and values in range; logs a manual_adjustment interaction
- get_interaction_stats() returns totals only — no UI currently uses this
- set_goal() loads predefined weight overrides from ANALYSIS_GOALS; uses .clear() + .update() for dict identity
- apply_temporal_decay() decays weights for interactions older than the median; 5 interactions decay one step

### Data Quality Pipeline (data_quality.py:59-170)

Pipeline ordering (critical — dedup/normalize names before column-level ops):

1. `_normalize_missing` — convert known missing tokens (NA, N/A, NULL, "", "-", "?", etc.) → NaN; gates on `is_object_dtype` or `is_string_dtype`
2. `_deduplicate_columns` — rename duplicate headers with `_1`, `_2` suffixes
3. `_normalize_column_names` — strip whitespace, lowercase, spaces→underscores, sanitize special chars
4. `_normalize_infinite` — replace inf/-inf with NaN
5. `_remove_empty_columns` — drop all-null columns
6. `_remove_empty_rows` — drop all-null rows
7. `_detect_sparse_columns` — flag columns >50% missing (`SPARSE_THRESHOLD`)
8. `_detect_constant_columns` — flag columns with ≤1 unique value
9. `_detect_mixed_types` — flag object/string-dtype columns with multiple Python types
10. `_infer_types` — cast numeric (90% confidence threshold) and datetime (name-hint + 80% parse ratio); gates on `is_object_dtype` or `is_string_dtype`

### QualityReport Schema (data_quality.py:33-46)
```python
@dataclass
class QualityReport:
    completeness: float              # 0–1 ratio of non-null cells
    uniqueness: float                # 0–1 avg ratio of unique/total per col
    datatype_consistency: dict        # col → {consistent: bool, note: str}
    duplicate_rows: int
    null_percentages: dict[str, float]
    memory_usage_mb: float
    overall_quality_score: float      # 0–1 composite from weighted sub-scores
    warnings: list[str]
    num_cols_after_cleanse: int
    row_count_after_cleanse: int
    rows_removed_fully_empty: int
    cols_removed_fully_empty: int
    sparse_columns: list[str]
    constant_columns: list[str]
    mixed_type_columns: list[str]
    duplicate_columns_renamed: list[tuple[str, str]]
```

### Quality Score Weights
| Component | Weight | Source |
|-----------|--------|--------|
| Completeness | 0.3 | Overall null ratio |
| Uniqueness | 0.2 | Avg distinct ratio per col |
| No Duplicate Rows | 0.15 | 1.0 − min(dup_count / row_count, 0.5) |
| No Sparse Columns | 0.15 | 1.0 − min(sparse_count / col_count, 0.5) |
| No Constant Columns | 0.10 | 1.0 − min(constant_count / col_count, 0.5) |
| No Mixed-Type Columns | 0.10 | 1.0 − min(mixed_count / col_count, 0.5) |

### Data Profile Schema (data_processor.py:32-48)
Profile is computed on the **post-cleanse** DataFrame. Column names are already normalized (lowercase, underscored).
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
- `explain_recommendation(rec, data_profile, user_prefs, quality_report)` → dict with 'reasons' and 'technique_reasons'
- `compare_recommendations(rec1, rec2)` → markdown table comparing scores and factors
- `global_explanation_summary(data_profile, quality_report, interaction_history, user_prefs)` → session-wide markdown summary
- `explain_user_preferences(preferences)` → plain-text summary

### NLQ Engine (nlq_engine.py)
- `match_query(text)` → (match_type, confidence, columns) tuple or (None, 0.0, [])
- `match_query_with_llm(text, columns, llm_classifier)` → (match_type, confidence, columns, source) — LLM primary, NLP fallback
- Uses NLP (tokenization, stemming, synonym expansion, TF-weighted overlap scoring) + optional LLM classifier
- `format_result(match, confidence, mentioned_columns)` → dict with title, description, techniques or None

### LLM Adapter (llm_adapter.py)
- Local (Ollama), OpenRouter, Groq, Custom API
- Streaming output via `_stream_response()`
- Context trimming (last 5 interactions) for speed
- Auto model selection for OpenAI-compatible endpoints
- LRU-capped AI cache at 20 entries (key format: `_ai_{provider}_{type}_{i}_{hash(tuple(columns))}`)
- Forced AI reload via button clears cache for current analysis
- `classify_nlq(query, columns)` — classifies a free-text query into an analysis type using LLM
- `chat_with_data(query, data_profile, df, conversation_history)` — answers free-form dataset questions with full dataset context

## Key Decisions (DO NOT REVERSE)

1. **No ML/AI** — PreferenceTracker uses fixed deltas, not learned models. UI must not claim "learning" or "AI."
2. **Session-only state** — preferences reset on page reload. No persistence layer (optional save/load to JSON is user-initiated).
3. **Plotly only** — matplotlib/seaborn/scipy removed from both code and requirements. All charts interactive.
4. **Pure pandas** — Polars not used. Keep it consistent.
5. **Empty dataset guard** — profile_dataset() returns empty schema dict when df.empty; load_data raises ValueError on unsupported format.
6. **No format inference fallback** — removed because it masked real errors with misleading openpyxl messages.
7. **Preserved dict identity** — PreferenceTracker.set_goal() uses .clear() + .update() instead of reassignment to keep st.session_state.user_preferences in sync.

## Critical Gotchas

- **Pipeline order matters** — `_deduplicate_columns` and `_normalize_column_names` must run BEFORE `_normalize_infinite` (which uses `select_dtypes`) and `_remove_empty_columns` (which checks by name). Duplicate columns cause `select_dtypes` to return a deduped frame that can't be assigned back.
- **Slider form pattern** — app.py uses `st.form()` for preferences with `st.form_submit_button()`. `set_preferences()` validates ALL required keys; missing any raises ValueError.
- **Toggle key uniqueness** — `'show_why_' + rec['type'] + '_' + str(i)` — the `str(i)` index is critical because two recommendations of the same type would otherwise share the same key.
- **Time series limitation** — plots against `df.select_dtypes(include=['number']).columns[0]` (first numerical col) only.
- **Correlation default** — selects first 5 numerical columns by default using `[:min(5, len(column_options))]`.
- **Categorical default** — selects first 3 low-cardinality columns (sorted by nunique ascending).
- **Interaction tracking** — column selection calls `preference_learner.track_interaction(rec, 'selected')` which permanently adjusts the weight. This means every multiselect change adds +0.05. Not debounced.
- **Outlier detection** — uses IQR (Q1-1.5×IQR, Q3+1.5×IQR). Reports percentage of rows flagged; does NOT return indices.
- **Diversity penalty** — penalizes second recommendation sharing same category. Categories: numerical, numerical_pairs, categorical, datetime, any. `any` (missing_values) never penalized.
- **Column interestingness** — columns within each recommendation sorted by `_score_column_interestingness()`. For numerical: skew × 0.6 + outlier% × 0.4. For categorical: peaks at cardinality ~10. Missing data penalizes score 50%.
- **Global summary** — only appears after first recommendation is rendered. Reads `st.session_state.interaction_history` directly.
- **Progressive sampling** — only triggers on file load. Uses `st.checkbox(key="sample_large_dataset")` inside the conditional `if` block. Session state persists the choice; checkbox only appears once per file.
- **Goal setting** — `set_goal()` mutates `preference_weights` in-place (`.clear()` + `.update()`) to preserve dict identity with `st.session_state.user_preferences`. Do NOT reassign.
- **Counterfactual slider** — uses inline score scaling (`new_score = old_score × (cf_value / old_pref)`), NOT full engine re-run. This avoids O(n²) recomputation.
- **AI cache** — LRU-capped at 20 entries using session state key scan + delete-oldest. Keys format: `_ai_{provider}_{type}_{i}_{hash(tuple(columns))}`.
- **Expert mode** — sidebar `st.toggle("_expert_mode")`. Shows raw DataFrame + CSV download + full recommendation JSON. Uses `st.caption` not nested expanders.
- **NLQ bar** — `nlq_engine.py` uses keyword patterns (no LLM). Import moved to top level to avoid per-rerun import overhead.
- **Input validation** — `data_processor.load_data()` reads uploaded file into `BytesIO` for size/empty checks before pandas parsing. Falls back to `latin1` encoding on `UnicodeDecodeError`.
- **Column rename safety** — re-cleansing after column rename uses `skip_name_normalization=True` to prevent `_normalize_column_names()` from undoing user renames.
- **Second project copy** — `C:\Users\Ashay\X-adapativeEDA\` is maintained in sync with `D:\Personal projects\X-adapativeEDA\`. Changes should be copied unless user says otherwise.
- **Pandas 3.0 string dtype** — pandas 3.0 uses `str` (StringDtype) instead of `object` for string columns. `is_object_dtype()` returns False for `str` dtype, so all `is_string_dtype()` checks must be added alongside `is_object_dtype()` in `_normalize_missing`, `_infer_types`, and `_detect_mixed_types`.
- **3-step pre-analysis pipeline** — after upload, user goes through: rename unnamed columns → drop first N rows → Finalize. The finalize block must also save `st.session_state.df = df` to persist the renamed/dropped DataFrame.
- **rec_columns tracking** — column selections per recommendation are tracked via `st.session_state._rec_cols_{type}_{i}` keys to preserve selection across reruns.
- **LLM as primary NLQ classifier** — When AI is enabled, `match_query_with_llm()` uses the LLM first via `classify_nlq()` and falls back to NLP only if the LLM is unreachable. Separate toggle `_chat_enabled` for the chat interface, independent of `ai_enabled`.
- **Chat context building** — `chat_with_data()` sends dataset stats (shape, columns, sample rows, missing %, skew, outliers) as context to the LLM. Keeps last 6 conversation exchanges for follow-ups. Provider settings shared with AI insights.
- **NLQ bar removed** — The `🔍 Ask about your data` text input was removed. Replaced by the dedicated `💬 Ask anything about your data` chat interface. The underlying NLQ engine still powers the NLP classifier for internal routing.

## File Loading (data_processor.py)
- Supports: CSV, XLSX, XLS, JSON
- Streamlit `file_uploader` accepts these four extensions
- Raises `ValueError` on unsupported format (no fallback chain)
- Raw df flows through `cleanse()` → cleaned df → `profile_dataset()` → profile dict

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
- Run all tests: `python test_phase1.py && python test_phase2.py && python test_phase3.py && python test_phase4.py && python test_data_quality.py`
- Manual verification: upload `.xlsx` file, confirm profile loads, click through all 6 analysis types, verify 👍/👎 adjustments stick in sidebar chart, test goal switching + save/load + decay + expert toggle + NLQ bar + 3-step pipeline
- Lint: none configured. No type annotations in most modules.

## Git
- Remote: `https://github.com/AshayK003/XadaptiveEDA.git` (branch: main)
- Init/initial commit message style: descriptive, lowercase prefix (e.g., "docs: rewrite README with module docs, data profile schema, analysis types, and limitations")
