ANALYSIS_TYPES = [
    'distribution',
    'correlation',
    'missing_values',
    'categorical',
    'outliers',
    'time_series'
]

DEFAULT_PREFERENCES = {
    'distribution': 0.5,
    'correlation': 0.5,
    'missing_values': 0.5,
    'categorical': 0.5,
    'outliers': 0.5,
    'time_series': 0.5
}

ANALYSIS_GOALS = {
    "General overview": {
        "distribution": 0.5, "correlation": 0.5, "missing_values": 0.5,
        "categorical": 0.5, "outliers": 0.5, "time_series": 0.5
    },
    "Explore distributions": {
        "distribution": 0.9, "correlation": 0.3, "missing_values": 0.3,
        "categorical": 0.3, "outliers": 0.8, "time_series": 0.3
    },
    "Find relationships": {
        "distribution": 0.3, "correlation": 0.9, "missing_values": 0.3,
        "categorical": 0.5, "outliers": 0.3, "time_series": 0.8
    },
    "Check data quality": {
        "distribution": 0.3, "correlation": 0.3, "missing_values": 0.9,
        "categorical": 0.7, "outliers": 0.5, "time_series": 0.3
    }
}

DECAY_HALF_LIFE_HOURS = 1.0
