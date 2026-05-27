# EDA Assistant

A Streamlit application that profiles datasets, scores six analysis types by relevance, and generates plotly visualizations. Analysis rankings incorporate both dataset characteristics and user-set priority weights.

## Quick Start

```bash
python -m venv venv
.\venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## How It Works

### Data Flow

```
Upload file → load_data() → raw df
                                ↓
                        cleanse() → cleaned df + QualityReport
                                ↓
                    profile_dataset() → profile dict
                                            ↓
        User priorities ──→ generate_recommendations() → ranked list
                                 ↓
              User selects analysis → VisualizationGenerator → plotly figure
                                 ↓
              User clicks 👍/👎 → PreferenceTracker → weights adjusted
```

### Scoring Formula

Each analysis type gets a score: `base_score × data_relevance × user_priority`

- **base_score**: fixed per type (0.6–0.9), reflects general usefulness
- **data_relevance**: computed from actual dataset characteristics (0.5–1.0)
- **user_priority**: slider value or feedback-adjusted weight (0.1–1.0)

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

Every uploaded file passes through a preprocessing pipeline that normalizes, validates, and reports on data quality before profiling and analysis.

### Pipeline Steps

| Step | Operation |
|------|-----------|
| Missing token normalization | `"NA", "N/A", "NULL", "", "-", "?", "#N/A"` → `NaN` |
| Duplicate column renaming | `col, col_1, col_2` suffixes |
| Column name normalization | lowercase, spaces→underscores, special chars sanitized |
| Infinite value replacement | `inf`/`-inf` → `NaN` |
| Empty column removal | Drop all-null columns |
| Empty row removal | Drop all-null rows |
| Sparse column detection | Columns >50% missing flagged |
| Constant column detection | Columns with ≤1 unique value flagged |
| Mixed-type detection | Object columns with multiple Python types flagged |
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
    overall_quality_score: float      # 0–1 composite (see weights below)
    warnings: list[str]
    sparse_columns: list[str]
    constant_columns: list[str]
    mixed_type_columns: list[str]
```

Quality score weights: completeness (0.3), uniqueness (0.2), no duplicates (0.15), no sparse (0.15), no constant (0.10), no mixed-types (0.10).

### UI Feedback

Warning banners appear above the data overview for:
- Sparse columns (>50% missing)
- Constant columns (single value)
- Mixed data types
- Duplicate rows
- Large memory usage

A "Data Quality Report" expander shows the full quality score, metrics, and detailed diagnostics.

## Priority Tracking

**`PreferenceTracker`** adjusts weights by fixed deltas:

| Action | Delta | Clamp |
|---|---|---|
| 👍 Useful | +0.10 | [0.1, 1.0] |
| Column selection | +0.05 | [0.1, 1.0] |
| 👎 Not Useful | −0.10 | [0.1, 1.0] |

All weights are session-only (lost on page reload). Manual slider overrides are logged but not decayed.

## Insight Generation

`insight_generator.py` produces text from actual data values — no pre-written templates. Examples:

- `"Dataset has 1,234 rows and 15 columns."`
- `"Missing values detected — 'age' has the most (40.2%)."`
- `"'revenue' is right-skewed (skewness=2.34)."`

## Visualization

All charts use **plotly** (interactive: zoom, pan, hover, download as PNG). No matplotlib or seaborn dependency.

Supported chart types by analysis:

| Analysis | Chart Types |
|---|---|
| Distribution | Histogram, Boxplot |
| Correlation | Heatmap (triangular + full) |
| Missing Values | Bar |
| Categorical | Bar, Pie |
| Outliers | Boxplot |
| Time Series | Line |

## Project Structure

```
├── app.py                    # Streamlit UI (sidebar, recommendations, viz)
├── data_processor.py         # File loading, dataset cleansing, profiling
├── data_quality.py           # DataQualityPipeline, QualityReport, cleanse()
├── recommendation_engine.py  # Scoring and ranking logic
├── preference_learner.py     # PreferenceTracker: feedback → weight adjustment
├── insight_generator.py      # Data-driven text generation
├── visualization_generator.py# Plotly chart creation
├── constants.py              # Shared type lists and default values
├── requirements.txt
└── README.md
```

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

- Session-only state: preferences reset on page reload
- No ML or AI: scoring uses fixed heuristics, not learned models
- Time series analysis plots against the first numerical column only
- Correlation uses pearson by default (spearman/kendall not exposed in UI)
- Large datasets (>100k rows) may be slow due to full in-memory processing
