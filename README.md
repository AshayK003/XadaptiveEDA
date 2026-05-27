# EDA Assistant

A Streamlit application that profiles datasets, scores six analysis types by data relevance × user priority, and generates plotly visualizations. Features a guided workflow (upload → rename → clean → finalize → analyze), AI-powered insights, preference tracking, data quality pipeline, and natural-language querying.

## Quick Start

```bash
python -m venv venv
.\venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## How It Works

### Workflow

```
Upload file → cleanse() → renamed/cleaned df
                                ↓
                      Profile dataset → profile dict
                                ↓
          User priorities → generate_recommendations() → ranked list
                                ↓
          User selects analysis → VisualizationGenerator → plotly figure
                                ↓
          User clicks 👍/👎 → PreferenceTracker → weights adjusted
```

The app introduces a **3-step pre-analysis pipeline**:
1. **Rename unnamed columns** — AI-suggested naming or manual entry
2. **Drop first N rows** — optional data cleanup
3. **Finalize** — triggers full profiling, quality report, and recommendation generation

### Scoring Formula

Each analysis type gets:  
`base_score × data_relevance × user_preference_score × quality_adjustment [× diversity_penalty]`

- **base_score**: fixed per type (0.6–0.9)
- **data_relevance**: computed from dataset characteristics (0.5–1.0)
- **user_preference_score**: slider value or feedback-adjusted weight (0.1–1.0)
- **quality_adjustment**: 0.5 + 0.5 × quality_score
- **diversity_penalty**: 0.85 if same-category duplicate, else 1.0

Results sorted descending, top 5 displayed.

## Six Analysis Types

| Type | Trigger | Default Chart |
|---|---|---|
| **Distribution** | Any numerical column | Histogram + Boxplot |
| **Correlation** | ≥2 numerical columns | Triangular + Full heatmap |
| **Missing Values** | Any null values | Bar chart of null % by column |
| **Categorical** | Any categorical column | Bar or Pie chart |
| **Outliers** | IQR-detectable outliers | Boxplot |
| **Time Series** | Date/time column detected | Line chart |

## Data Profile Structure

`DataProcessor.profile_dataset(df)` returns a dict with these keys:

| Key | Type | Description |
|---|---|---|
| `shape` | `(int, int)` | Row and column count |
| `dtypes` | `dict[str, str]` | Column name → dtype string |
| `missing_values` | `dict[str, int]` | Column name → null count |
| `missing_percentage` | `dict[str, float]` | Column name → null % |
| `numerical_cols` | `list[str]` | Numerical column names |
| `categorical_cols` | `list[str]` | Object/category column names |
| `unique_counts` | `dict[str, int]` | Column name → nunique() |
| `skewness` | `dict[str, float\|None]` | Column name → skew() |
| `correlation_exists` | `bool` | `len(numerical_cols) > 1` |
| `time_series_candidates` | `list[str]` | Columns matching date patterns |
| `categorical_cardinality` | `dict[str, int]` | Categorical column → nunique() |
| `has_outliers` | `dict[str, float]` | Column → % rows beyond 1.5×IQR |

## Data Quality Pipeline

Every uploaded file passes through a preprocessing pipeline that normalizes, validates, and reports on data quality.

### Pipeline Steps (order-sensitive)

| Step | Operation |
|---|---|
| Missing token normalization | `"NA", "N/A", "NULL", "", "-", "?", "#N/A"` → `NaN` |
| Duplicate column renaming | `col, col_1, col_2` suffixes |
| Column name normalization | lowercase, spaces→underscores, special chars sanitized |
| Infinite value replacement | `inf`/`-inf` → `NaN` |
| Empty column removal | Drop all-null columns |
| Empty row removal | Drop all-null rows |
| Sparse column detection | Columns >50% missing flagged |
| Constant column detection | Columns with ≤1 unique value flagged |
| Mixed-type detection | Object/string columns with multiple Python types flagged |
| Type inference | Numeric cast at 90% confidence; datetime on name hints |

### QualityReport

```python
@dataclass
class QualityReport:
    completeness: float              # 0–1 non-null cell ratio
    uniqueness: float                # 0–1 avg unique/total per column
    datatype_consistency: dict       # col → {consistent, note}
    duplicate_rows: int
    null_percentages: dict[str, float]
    memory_usage_mb: float
    overall_quality_score: float      # 0–1 composite
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

Quality score weights: completeness (0.3), uniqueness (0.2), no duplicates (0.15), no sparse (0.15), no constant (0.10), no mixed-types (0.10).

## Priority Tracking

**PreferenceTracker** adjusts weights by fixed deltas (no ML):

| Action | Delta | Clamp |
|---|---|---|
| 👍 Useful | +0.10 | [0.1, 1.0] |
| Clicked an analysis | +0.05 | [0.1, 1.0] |
| Column selection | +0.05 | [0.1, 1.0] |
| 👎 Not Useful | −0.10 | [0.1, 1.0] |
| Ignored (not clicked) | −0.02 | [0.1, 1.0] |

### Additional Features

- **Analysis Goals** — set a focus (e.g., "Find Anomalies") that overrides all priorities via goal-specific weights
- **Save/Load** — export preferences to `~/.eda_assistant_prefs.json`
- **Temporal Decay** — older interactions lose influence over time; 5 interactions decay one step each

## AI Insights (Optional)

An optional LLM layer enriches analysis with natural-language observations. Supports three modes:

| Provider | Key Requirement | Default Model |
|---|---|---|
| Local (Ollama) | None | llama3.2:3b |
| OpenRouter (free tier) | `OPENROUTER_API_KEY` | meta-llama/llama-3.2-3b-instruct |
| Groq (free tier) | `GROQ_API_KEY` | llama-3.2-3b-preview |
| Custom API | `CUSTOM_API_KEY` + endpoint | Configurable |

Toggle in sidebar. Features streaming output, context trimming for speed, automatic model selection based on OpenAI-compatibility, and an LRU-cached response store (capped at 20 entries).

## Natural Language Query Bar

Type freeform queries like `"show me outliers"` or `"correlation between columns"` into the NLQ bar (visible after finalization). Uses keyword pattern matching (no LLM) to map queries to analysis types with confidence scores.

## Insight Generation

`insight_generator.py` produces text from actual data values — no pre-written templates. Includes:

- `generate_insights(data_profile)` — per-column observations
- `explain_recommendation(rec, data_profile, user_prefs, quality_report)` — why a recommendation scored as it did
- `compare_recommendations(rec1, rec2)` — markdown table comparing scores and factors
- `global_explanation_summary(...)` — session-wide markdown summary
- `explain_user_preferences(preferences)` — plain-text preference summary

## Counterfactual Slider

Each recommendation's explanation includes a "What if?" slider that scales the score in-place (no full engine re-run) to show how changing the priority would affect ranking.

## Visualization

All charts use **plotly** (interactive: zoom, pan, hover, download as PNG). No matplotlib or seaborn.

| Analysis | Chart Types |
|---|---|
| Distribution | Histogram, Boxplot |
| Correlation | Heatmap (triangular + full) |
| Missing Values | Bar |
| Categorical | Bar, Pie |
| Outliers | Boxplot |
| Time Series | Line |

## Expert Mode

Sidebar toggle that reveals: raw DataFrame viewer, CSV download button, and full recommendation JSON with technical details (base score, data relevance, quality adjustment, diversity penalty).

## Project Structure

```
├── app.py                    # Streamlit UI (sidebar, recommendations, viz)
├── data_processor.py         # File loading, dataset cleansing, profiling
├── data_quality.py           # DataQualityPipeline, QualityReport, cleanse()
├── recommendation_engine.py  # Scoring/ranking, diversity, column interestingness
├── preference_learner.py     # PreferenceTracker: 5 actions + goal/decay/persistence
├── insight_generator.py      # Explanations, comparisons, global summary
├── visualization_generator.py# Plotly chart creation
├── constants.py              # ANALYSIS_TYPES, DEFAULT_PREFERENCES, ANALYSIS_GOALS
├── llm_adapter.py            # Optional LLM analysis (local + remote)
├── nlq_engine.py             # Natural language query → analysis type
├── requirements.txt
└── README.md
```

## Testing

Run all 42 tests:
```bash
python test_phase1.py && python test_phase2.py && python test_phase3.py && python test_phase4.py && python test_data_quality.py
```

Test coverage: diversity penalty, implicit tracking, comparative explanations, regression checks, column interestingness, global summary, empty history, sampling concept, goal setting, temporal decay, save/load round-trip, NLQ keyword matching, data quality pipeline (12 pipeline steps).

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | ≥1.36 | Web UI framework |
| pandas | ≥1.5 | Data processing |
| numpy | ≥1.23 | Numerical operations |
| plotly | ≥5.10 | Interactive charts |
| openpyxl | ≥3.1 | Excel (.xlsx) reading |
| xlrd | ≥2.0 | Legacy Excel (.xls) reading |

## Limitations

- Session-only state: preferences reset on page reload (optional save/load via JSON)
- No ML or AI: scoring uses fixed heuristics, not learned models; LLM insights are optional
- Time series analysis plots against the first numerical column only
- Correlation uses pearson by default (spearman/kendall not exposed in UI)
- Large datasets (>100k rows) may be slow (progressive sampling available as opt-in)
- Column selection changes permanently adjust preference weights (not debounced)
