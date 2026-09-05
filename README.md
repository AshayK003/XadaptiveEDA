<div align="center">

# X-Adaptive EDA

### Explore. Adapt. Understand.

**An adaptive data analysis tool that learns your priorities and recommends the most relevant analyses.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-≥1.36-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen.svg)](#testing)
[![pandas](https://img.shields.io/badge/pandas-≥1.5-15045D.svg)](https://pandas.pydata.org/)
[![GitHub Stars](https://img.shields.io/github/stars/AshayK003/XadaptiveEDA?logo=github)](https://github.com/AshayK003/XadaptiveEDA)

[Quick Start](#quick-start) • [Features](#features) • [Demo](#demo) • [Architecture](#architecture) • [Contributing](#contributing)

</div>

---

## What is X-Adaptive EDA?

X-Adaptive EDA is a Streamlit-based exploratory data analysis tool that goes beyond static reporting. It **adapts to how you work** — learning from your feedback, prioritizing what matters to you, and explaining why each recommendation scored the way it did.

**Upload a dataset → Get intelligent recommendations → Explore with interactive charts → Chat with your data → Your preferences evolve as you go.**

---

## Demo

<!-- Add screenshots here -->
<!-- 
![Main Interface](screenshots/main.png)
![Recommendations](screenshots/recommendations.png)
![Chat Interface](screenshots/chat.png)
![Quality Report](screenshots/quality.png)
-->

*Screenshots coming soon. Run locally to see the full experience.*

---

## Features

### Core Analytics
- **8 Analysis Types** — Distribution, Correlation, Missing Values, Categorical, Outliers, Time Series, Clustering, Feature Importance
- **Adaptive Scoring** — Recommendations learn from your feedback and adjust in real-time
- **Explainable Recommendations** — Every score decomposes into its components with confidence intervals
- **Interactive Visualizations** — Plotly charts with zoom, pan, hover, and download

### Intelligence
- **AI-Powered Insights** — LLM-generated observations for each analysis (Ollama, OpenRouter, Groq, or Custom API)
- **Chat with Your Data** — Ask natural language questions about your dataset
- **Smart Column Naming** — AI suggests names for unnamed columns
- **NLQ Classifier** — Understands queries like "show me outliers in revenue"

### Adaptation
- **Preference Tracking** — 👍/👎 feedback permanently adjusts analysis priorities
- **Temporal Decay** — Older preferences fade over time
- **Novelty Dampening** — Avoids repeating the same analyses
- **Column Affinity** — Boosts analyses involving columns you frequently explore
- **ε-Greedy Exploration** — Occasionally shows unexpected analyses to discover new insights

### Data Quality
- **10-Step Quality Pipeline** — Normalizes, deduplicates, infers types, and scores your data
- **Per-Row Outlier Explainability** — See which column triggered each outlier and why
- **Progressive Sampling** — Large datasets (>50k rows) offer stratified sampling to ~10k

### Developer Experience
- **Session Persistence** — Save/load via SQLite
- **68 Tests** — Comprehensive test suite
- **Rate Limiting** — Remote API calls capped at 10/minute
- **GPU Acceleration** — Ollama auto-uses GPU with CPU fallback

---

## Why This Project Exists

Most EDA tools give you static reports. X-Adaptive EDA does three things differently:

1. **It learns** — Every 👍/👎 shifts future recommendations toward what you care about
2. **It explains** — No black boxes. Every score shows its formula. Counterfactual sliders let you ask "what if?"
3. **It adapts in real-time** — No waiting for retraining. Feedback takes effect immediately.

This makes it ideal for:
- **Data scientists** doing exploratory analysis
- **Analysts** who need quick, relevant insights
- **Students** learning data analysis
- **Teams** exploring unfamiliar datasets

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Streamlit (≥1.36) |
| **Data** | pandas, NumPy |
| **Visualization** | Plotly |
| **LLM** | Ollama (local), OpenRouter, Groq, Custom API |
| **NLP** | Custom tokenizer + stemmer (no external deps) |
| **Persistence** | SQLite, JSON |
| **Testing** | pytest-compatible test files |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                │
│  Sidebar: Dataset • Priorities • AI • Sessions          │
│  Main: Recommendations • Visualizations • Chat          │
└─────────┬───────────────────────────────────┬───────────┘
          │                                   │
    ┌─────▼─────┐                     ┌───────▼───────┐
    │   Data     │                     │  Recommendation│
    │  Processor │                     │    Engine      │
    │  + Quality │                     │  (scoring,     │
    │  Pipeline  │                     │   ranking)     │
    └─────┬─────┘                     └───────┬───────┘
          │                                   │
    ┌─────▼─────┐                     ┌───────▼───────┐
    │    LLM     │                     │   Preference  │
    │   Adapter  │                     │    Tracker    │
    │  (insights, │                     │  (adaptation) │
    │   chat)    │                     └───────────────┘
    └───────────┘
```

### Data Flow

```
Upload → Cleanse → Profile → Score → Rank → Visualize → Feedback → Adapt
                ↓                              ↑
          Quality Report              Counterfactual Slider
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- (Optional) [Ollama](https://ollama.ai/) for local LLM features

### Installation

```bash
# Clone the repository
git clone https://github.com/AshayK003/XadaptiveEDA.git
cd XadaptiveEDA

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Environment Setup (Optional)

For LLM features, copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env with your keys
```

**No API key needed for local Ollama** — just install and run.

---

## Usage

### Basic Workflow

1. **Upload** a CSV, Excel, or JSON file
2. **Rename** unnamed columns (AI suggestions or manual)
3. **Finalize** to generate the full analysis
4. **Explore** recommended analyses ranked by relevance
5. **Give feedback** (👍/👎) to refine future recommendations
6. **Chat** with your data in natural language

### Example Session

```python
# The app runs via Streamlit — no Python code needed
# Just run:
streamlit run app.py

# Then in the browser:
# 1. Upload sales_data.csv
# 2. Click "Finalize Dataset"
# 3. Click 👍 on "Distribution Analysis"
# 4. Ask: "What's the correlation between price and quantity?"
```

### Expert Mode

Toggle **Dev Mode** in the sidebar to reveal:
- Raw DataFrame viewer
- CSV download button
- Full recommendation JSON with all scoring components

---

## Configuration

### Analysis Goals

Choose a preset goal to automatically weight analysis types:

| Goal | Distribution | Correlation | Missing | Categorical | Outliers | Time Series | Clustering | Feature Imp |
|------|-------------|-------------|---------|-------------|----------|-------------|------------|-------------|
| General | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| Distributions | 0.9 | 0.3 | 0.3 | 0.3 | 0.8 | 0.3 | 0.3 | 0.4 |
| Relationships | 0.3 | 0.9 | 0.3 | 0.5 | 0.3 | 0.8 | 0.6 | 0.7 |
| Data Quality | 0.3 | 0.3 | 0.9 | 0.7 | 0.5 | 0.3 | 0.3 | 0.5 |

### Scoring Formula

```
final_score = base_score × data_relevance × user_pref × quality_adj
            × diversity_penalty × novelty_penalty × avoidance_penalty × affinity_boost
```

All multipliers are documented in `recommendation_engine.py`.

### LLM Providers

| Provider | Key | Default Model | Rate Limit |
|----------|-----|---------------|------------|
| Local (Ollama) | None | qwen2.5-coder:7b | Unlimited |
| OpenRouter | `OPENROUTER_API_KEY` | qwen/qwen2.5-7b-instruct | 10/60s |
| Groq | `GROQ_API_KEY` | llama-3.3-70b-versatile | 10/60s |
| Custom | `CUSTOM_API_KEY` + endpoint | Configurable | 10/60s |

---

## Project Structure

```
x-adaptive-eda/
├── app.py                    # Streamlit UI (orchestration, ~970 lines)
├── data_processor.py         # File loading, cleansing, profiling
├── data_quality.py           # 10-step quality pipeline, QualityReport
├── recommendation_engine.py  # Scoring, ranking, penalties, bootstrap CI
├── preference_learner.py     # Fixed-delta adaptation, goals, decay
├── insight_generator.py      # Explainable recommendations, comparisons
├── visualization_generator.py# Plotly charts (8 types, k-means, MI)
├── constants.py              # Analysis types, preferences, goals
├── llm_adapter.py            # LLM integration, rate limiting, chat
├── nlq_engine.py             # NLP query classifier (no external deps)
├── session_persistence.py    # SQLite save/load for sessions
├── requirements.txt          # 6 dependencies
├── .env.example              # Environment variable template
├── LICENSE                   # MIT License
├── README.md                 # This file
├── test_phase1.py            # Legacy smoke scripts (run via tests/)
├── test_phase2.py            # (kept for manual runs; CI uses tests/)
├── test_phase3.py            #
├── test_phase4.py            #
├── test_data_quality.py      #
├── test_session_persistence.py #
├── test_rate_limit.py        #
├── tests/                    # Real pytest suite (smoke + regression)
```

---

## Development Setup

```bash
# Install in development mode
pip install -r requirements.txt

# Run tests
python test_phase1.py && python test_phase2.py && python test_phase3.py && python test_phase4.py && python test_data_quality.py && python test_session_persistence.py && python test_rate_limit.py

# Run the app
streamlit run app.py
```

### Code Style

- snake_case for functions/variables
- PascalCase for classes
- Docstrings on all public functions
- Structured logging via `logging.getLogger(__name__)`
- No `print()` in source files (only in tests)

---

## Testing

**18 pytest tests** in `tests/` (plus the 7 legacy scripts they execute):

| File | Tests | Coverage |
|------|-------|----------|
| tests/test_smoke_scripts.py | 8 | Every legacy script exits 0 |
| tests/test_audit_fixes.py | 10 | Outlier scale, epsilon boost, dedup, decay, kmeans, heatmap, stemmer, endpoint guard, error redaction, score display |

Legacy print-scripts (`test_phase*.py`, etc.) are kept for manual runs;
CI executes the real suite:

```bash
# Run the suite
python -m pytest tests/ -q
```

---

## Roadmap

### Completed
- [x] 8 analysis types with adaptive scoring
- [x] Explainable recommendations with score decomposition
- [x] Interactive Plotly visualizations
- [x] LLM integration (Ollama, OpenRouter, Groq, Custom)
- [x] Chat with your data
- [x] Session persistence (SQLite)
- [x] ε-greedy exploration
- [x] Rate limiting for remote APIs
- [x] 68 tests passing
- [x] MIT License

### Planned
- [ ] Spearman/Kendall correlation options
- [ ] Custom k-means cluster count slider
- [ ] Export analysis report as PDF/HTML
- [ ] Multi-dataset comparison
- [ ] Dashboard mode (persistent charts)
- [ ] Plugin system for custom analysis types
- [ ] Collaborative sessions (multi-user)

---

## Contributing

Contributions welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Guidelines

- Follow existing code style
- Add tests for new features
- Update README if needed
- Keep PRs focused (one feature per PR)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Streamlit](https://streamlit.io/) — Web framework
- [Plotly](https://plotly.com/python/) — Interactive visualizations
- [Ollama](https://ollama.ai/) — Local LLM hosting
- [pandas](https://pandas.pydata.org/) — Data manipulation

---

## FAQ

**Q: Do I need an API key to use this?**
A: No. Local Ollama works without any API keys. API keys are only needed for OpenRouter, Groq, or Custom API providers.

**Q: What file formats are supported?**
A: CSV, XLSX, XLS, and JSON files up to ~50 MB.

**Q: How does the adaptation work?**
A: Fixed-delta adjustments (not ML). 👍 adds +0.10, 👎 subtracts -0.10, column selection adds +0.03. All weights stay in [0.1, 1.0].

**Q: Can I save my session?**
A: Yes. Click "Save Session" in the sidebar. Sessions persist in SQLite at `~/.eda_assistant_sessions.db`.

**Q: How accurate are the AI insights?**
A: Insights are generated from your actual data values — no pre-written templates. Quality depends on the LLM provider and model used.

**Q: Is my data sent to external servers?**
A: Only if you use OpenRouter, Groq, or Custom API. Local Ollama keeps everything on your machine.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| App won't start | Check Python version (3.10+), run `pip install -r requirements.txt` |
| Ollama not reachable | Run `ollama serve` in a terminal |
| GPU not detected | Install NVIDIA drivers, restart Ollama |
| Slow LLM responses | Use CPU mode: `set OLLAMA_NUM_GPU=0` before starting Ollama |
| Large file warning | Files >50 MB may be slow; use sampling for datasets >50k rows |
| Import errors | Ensure virtual environment is activated |

---

## Security

- API keys stored in `.env` (gitignored)
- No hardcoded secrets in source code
- Parameterized SQL queries (no injection risk)
- Local Ollama keeps data on your machine
- Remote API calls rate-limited to 10/60s

---

<div align="center">

**Built with ❤️ for the data community**

[⭐ Star this repo](https://github.com/AshayK003/XadaptiveEDA) • [🐛 Report Bug](https://github.com/AshayK003/XadaptiveEDA/issues) • [💡 Request Feature](https://github.com/AshayK003/XadaptiveEDA/issues) • [☕ Support the developer](https://chai4.me/ashaykushwaha003)

</div>
