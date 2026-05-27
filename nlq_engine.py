"""Natural language query engine for analysis type recommendation."""

import re

QUERY_PATTERNS = [
    (r'(distribut|histogram|density|kde|shape|spread|range|min|max)', 'distribution'),
    (r'(correlat|relationship|pairwise|heatmap|multicol|multivariate|scatter)', 'correlation'),
    (r'(missing|null|nan|empty|incomplete|blank|gaps|sparse)', 'missing_values'),
    (r'(categor|bar chart|frequency|count|enum|nominal|label|segment)', 'categorical'),
    (r'(outlier|anomal|extreme|abnormal|unusual|rare|suspicious|z.score|iqr)', 'outliers'),
    (r'(time|trend|season|forecast|series|date|temporal|over time|chrono)', 'time_series'),
]


def match_query(query):
    """Match a natural language query to an analysis type.
    
    Returns (analysis_type, confidence) tuple, or (None, 0) if no match.
    Uses keyword pattern matching.
    """
    query_lower = query.lower().strip()
    return _keyword_match(query_lower)


def _keyword_match(query_lower):
    """Match query against keyword patterns."""
    for pattern, analysis_type in QUERY_PATTERNS:
        if re.search(pattern, query_lower):
            return (analysis_type, 0.7)
    return (None, 0)


def format_result(analysis_type, confidence):
    """Format match result as a display-friendly string."""
    if analysis_type is None:
        return None
    names = {
        'distribution': 'Distribution Analysis',
        'correlation': 'Correlation Analysis',
        'missing_values': 'Missing Values Analysis',
        'categorical': 'Categorical Analysis',
        'outliers': 'Outlier Detection',
        'time_series': 'Time Series Analysis',
    }
    return {
        'type': analysis_type,
        'title': names.get(analysis_type, analysis_type),
        'confidence': confidence,
    }
