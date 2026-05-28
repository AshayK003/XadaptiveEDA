# X-Adaptive EDA

> **Explore. Adapt. Understand.**

An adaptive Streamlit application that profiles datasets, scores **eight** analysis types by data relevance × user priority, and generates plotly visualizations. Features a guided workflow (upload → rename → clean → finalize → analyze), AI-powered insights, a chat interface for asking any question about your dataset via LLM, preference tracking with temporal decay, a data quality pipeline, session persistence via SQLite, ε-greedy exploration, rate-limited remote LLM calls, and an NLP query classifier.

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
1. **Rename unnamed columns** — AI-suggested naming (LLM on by default) or manual entry
2. **Drop first N rows** — optional data cleanup
3. **Finalize** — triggers full profiling, quality report, and recommendation generation

### Scoring Formula

Each analysis type gets:  
`base_score × data_relevance × user_preference_score × quality_adjustment [× diversity_penalty] [× novelty_penalty] [× avoidance_penalty] [× affinity_boost]`

- **base_score**: fixed per type (0.6–0.9)
- **data_relevance**: computed from dataset characteristics (0.5–1.0), includes pairwise correlation for correlation type
- **user_preference_score**: slider value or feedback-adjusted weight (0.1–1.0)
- **quality_adjustment**: 0.5 + 0.5 × quality_score
- **diversity_penalty**: 0.85 if same-category duplicate, else 1.0
- **novelty_penalty**: 0.6 if exact combo viewed, 0.85 if same type
- **avoidance_penalty**: 0.5 if skipped ≥3 times, gradual for 1–2
- **affinity_boost**: up to 1.10× based on column co-occurrence Jaccard
- **confidence interval**: 95% CI via bootstrap (50 iterations) reported as `score_ci_lower` / `score_ci_upper`

Results sorted descending, top 5 displayed (with ε-greedy exploration support).

## Eight Analysis Types

| Type | Trigger | Default Chart |
|---|---|---|
| **Distribution** | Any numerical column | Histogram + Boxplot |
| **Correlation** | ≥2 numerical columns | Triangular + Full heatmap |
| **Missing Values** | Any null values | Bar chart of null % by column |
| **Categorical** | Any categorical column | Bar or Pie chart |
| **Outliers** | IQR-detectable outliers | Boxplot |
| **Time Series** | Date/time column detected | Line chart |
| **Clustering** | ≥2 numerical columns | K-means scatter (SVD 2D if >2 dims) |
| **Feature Importance** | ≥2 numerical columns | Mutual information bar chart |

## Chat with Your Data (LLM)

After finalization, toggle **"Ask anything about your data"** in the sidebar to open a chat interface. Ask free-form questions about your dataset:

- *"What's the average value of column X?"*
- *"Which columns have the most missing data?"*
- *"Are there any outliers in revenue?"*
- *"What's the correlation between duration and score?"*
- *"Tell me about the distribution of age"*

The chat sends dataset context (shape, column stats, sample rows, missing values, outliers, skewness) to the LLM. Conversation history (last 6 exchanges) is maintained for follow-ups. Works with all providers (Ollama, OpenRouter, Groq, Custom API).

## AI Insights (Optional)

An optional LLM layer on each analysis provides natural-language observations. AI is **enabled by default** (for column naming and insights).

| Provider | Key Requirement | Default Model |
|---|---|---|
| Local (Ollama) | None | qwen2.5-coder:7b |
| OpenRouter (free tier) | `OPENROUTER_API_KEY` | qwen/qwen2.5-7b-instruct |
| Groq (free tier) | `GROQ_API_KEY` | llama-3.3-70b-versatile |
| Custom API | `CUSTOM_API_KEY` + endpoint | Configurable |

Local Ollama automatically uses **GPU acceleration** when VRAM is available, with seamless **CPU fallback** for layers that don't fit in GPU memory. No manual configuration required. Remote API calls are **rate-limited** to 10 requests per 60 seconds. Features LRU-cached response store (capped at 20 entries).

## Session Persistence

Sessions can be saved to and loaded from a local SQLite database (`~/.eda_assistant_sessions.db`). Persists preferences, interaction history, active goal, last filename, and profile JSON. The actual DataFrame is not persisted — users re-upload the file. Recent sessions are listed with delete buttons.

## ε-Greedy Exploration

A sidebar toggle enables exploration mode. When active, there is a 10% chance per recommendation re-rank to swap one top-5 recommendation with a randomly chosen lower-ranked alternative, helping discover unexpected insights.

## Per-Row Outlier Explainability

For outlier analysis, a "Show outlier rows" checkbox reveals which columns triggered each row's outlier flag and how far beyond the IQR bound the value lies.

## Progressive Stratified Sampling

Datasets larger than 50,000 rows offer an opt-in checkbox to sample down to ~10,000 rows using stratified sampling (preserving categorical distributions). Falls back to random sampling if no suitable stratification column exists.

## NLP Query Classifier

The `nlq_engine.py` module classifies short queries into analysis types using tokenization, stemming, synonym expansion, and TF-weighted overlap scoring. Extracts column names from queries. (Legacy interface — the NLQ bar has been replaced by the chat interface; `match_query` and `format_result` remain as test-covered utilities.)

## Insight Generation

`insight_generator.py` produces text from actual data values — no pre-written templates. Includes:

- `explain_recommendation(rec, data_profile, user_prefs, quality_report)` — why a recommendation scored as it did
- `compare_recommendations(rec1, rec2)` — markdown table comparing scores and factors
- `global_explanation_summary(...)` — session-wide markdown summary

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
| Clustering | K-means scatter (numpy-only) |
| Feature Importance | Mutual information bar |

## Expert Mode

Sidebar toggle that reveals: raw DataFrame viewer, CSV download button, and full recommendation JSON with technical details (base score, data relevance, quality adjustment, diversity penalty, novelty penalty, avoidance penalty, affinity boost, confidence interval).

## Project Structure

```
├── app.py                    # Streamlit UI (sidebar, recommendations, viz, chat)
├── data_processor.py         # File loading, dataset cleansing, profiling
├── data_quality.py           # DataQualityPipeline, QualityReport, cleanse()
├── recommendation_engine.py  # Scoring/ranking, penalties, boosts, bootstrap
├── preference_learner.py     # PreferenceTracker: 5 actions + goal/decay/persistence
├── insight_generator.py      # Explanations, comparisons, global summary
├── visualization_generator.py# Plotly chart creation (incl. k-means, mutual info)
├── constants.py              # ANALYSIS_TYPES, DEFAULT_PREFERENCES, ANALYSIS_GOALS
├── llm_adapter.py            # LLM analysis, chat, column naming, rate limiting
├── nlq_engine.py             # NLP query classifier (stemming, synonyms, TF scoring)
├── session_persistence.py    # SQLite save/load for sessions
├── test_phase1.py–test_phase4.py  # 54 core unit tests
├── test_data_quality.py      # Data quality pipeline tests (12 tests)
├── test_session_persistence.py # SQLite session persistence tests (7 tests)
├── test_rate_limit.py         # LLM rate limiting tests (5 tests)
├── requirements.txt
└── README.md
```

## Testing

Run all tests:
```bash
python test_phase1.py && python test_phase2.py && python test_phase3.py && python test_phase4.py && python test_data_quality.py && python test_session_persistence.py && python test_rate_limit.py
```

68 tests covering: diversity penalty, implicit tracking, comparative explanations, regression checks, column interestingness, global summary, goal setting, temporal decay, save/load round-trip, NLQ via NLP (stemming, synonyms, TF scoring, column extraction), data quality pipeline (12 pipeline steps), SQLite session persistence (7 tests), LLM rate limiting (5 tests).

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

- Session-only state: preferences reset on page reload (optional save/load via SQLite)
- No ML or AI: scoring uses fixed heuristics, not learned models; LLM insights and chat are optional
- Time series analysis plots against the first numerical column only
- Correlation uses pearson by default (spearman/kendall not exposed in UI)
- Large datasets (>100k rows) may be slow (progressive sampling available as opt-in)
- Column selection changes permanently adjust preference weights (not debounced)
- Clustering uses a custom numpy-only k-means (not scikit-learn) — suitable for exploration, not production
- Remote API rate-limited to 10 calls/minute; local Ollama unlimited
- pandas 3.0 compatibility: categorical profiling, mixed-type detection, and stratified sampling handle `StringDtype` correctly
