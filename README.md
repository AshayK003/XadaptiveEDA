# Exploratory Data Analysis Assistant

A Streamlit tool that profiles datasets and recommends analyses based on data characteristics and user-set priorities.

## Overview

Upload a CSV, Excel, or JSON file. The tool profiles your data (column types, missing values, outliers, skewness), ranks six analysis types by relevance, and lets you generate visualizations. You can adjust priority sliders or give thumbs-up/down to shift future recommendations.

## Features

- Automatic data profiling (types, missing values, outliers, skewness, dates)
- Analysis recommendations ranked by data relevance × your priorities
- Manual priority adjustment via sliders
- Visualizations for distribution, correlation, missing values, categorical, outliers, time series
- Priority tracking across your session (resets on page reload)
- Supports CSV, Excel, JSON

## Installation

```
python -m venv venv
.\venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Usage

```
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## How It Works

### 1. Data Profiling

On upload, the tool scans the dataset and records:
- Numerical vs categorical column types
- Missing value counts and percentages
- Outlier detection via IQR
- Distribution skewness
- Date/time column candidates

### 2. Recommendation Scoring

Each of six analysis types gets a score: `base_score × data_relevance × user_priority`. Results are ranked highest to lowest.

### 3. Priority Tracking

Your interactions (likes, dislikes, column selections) adjust priority weights up or down. You can also set priorities directly with sliders. This is session-only — nothing is saved to disk.

### 4. Visualization

Select columns and generate charts using matplotlib/seaborn or plotly. Available chart types depend on the analysis.

## System Components

| Module | Purpose |
|--------|---------|
| `data_processor.py` | File parsing, dataset profiling |
| `recommendation_engine.py` | Scores and ranks analyses |
| `preference_tracker.py` | Adjusts weights based on feedback |
| `insight_generator.py` | Generates text from actual data values |
| `visualization_generator.py` | Creates matplotlib/plotly charts |

## License

MIT
