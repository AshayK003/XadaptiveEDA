"""Natural language query engine using NLP techniques.

Tokenizes, stems, expands synonyms, scores by token overlap,
and optionally extracts column names from queries.
"""

import math
import re

# ── Stemmer (rule-based, no deps) ─────────────────────────────

_STEMS = {
    'distributions': 'distribution', 'categories': 'category',
    'relationships': 'relationship', 'correlations': 'correlation',
    'outliers': 'outlier', 'anomalies': 'anomaly',
    'missing': 'missing', 'analyses': 'analysis',
    'series': 'series', 'values': 'value', 'columns': 'column',
    'features': 'feature', 'variables': 'variable', 'trends': 'trend',
    'patterns': 'pattern', 'histograms': 'histogram',
    'heatmaps': 'heatmap', 'boxplots': 'boxplot',
    'scatters': 'scatter', 'densities': 'density',
    'frequencies': 'frequency', 'segments': 'segment',
    'classes': 'class', 'labels': 'label', 'groups': 'group',
    'gaps': 'gap', 'blanks': 'blank', 'spikes': 'spike',
    'deviation': 'deviate', 'deviations': 'deviate',
    'extremes': 'extreme', 'unusual': 'unusual',
}

_SUFFIX_RULES = [
    (r'ies$', 'y'), (r'ves$', 'f'), (r'ing$', ''),
    (r'ions$', 'ion'), (r'ion$', 'ion'), (r'ed$', ''),
    (r'ly$', ''), (r'es$', ''), (r's$', ''),
]


def _stem(word):
    if len(word) < 3:
        return word
    word = word.lower()
    if word in _STEMS:
        return _STEMS[word]
    for pattern, replacement in _SUFFIX_RULES:
        if re.search(pattern, word):
            candidate = re.sub(pattern, replacement, word)
            if len(candidate) >= 3:
                return candidate
    return word


# ── Synonyms ──────────────────────────────────────────────────

_SYNONYMS = {
    'show': {'view', 'display', 'see', 'find', 'get', 'visualize',
             'plot', 'chart', 'give', 'reveal', 'render'},
    'find': {'show', 'view', 'display', 'see', 'get', 'locate',
             'detect', 'discover'},
    'analyze': {'analyse', 'examine', 'explore', 'investigate',
                'study', 'review', 'inspect', 'check'},
    'relationship': {'correlation', 'association', 'relation',
                     'link', 'connection', 'tie', 'dependency'},
    'correlation': {'relationship', 'association', 'relation',
                    'link', 'connection', 'pairwise', 'multivariate'},
    'distribution': {'spread', 'range', 'histogram', 'density',
                     'shape', 'dispersion', 'variance'},
    'histogram': {'distribution', 'frequency', 'bin', 'density'},
    'outlier': {'anomaly', 'extreme', 'abnormal', 'unusual',
                'rare', 'suspicious', 'spike', 'deviation',
                'aberrant', 'exceptional'},
    'anomaly': {'outlier', 'abnormal', 'unusual', 'rare',
                'suspicious', 'spike'},
    'missing': {'null', 'empty', 'blank', 'incomplete', 'gap',
                'sparse', 'nan', 'na', 'absent', 'void'},
    'categorical': {'category', 'segment', 'group', 'class',
                    'label', 'nominal', 'enum', 'type', 'bar',
                    'discrete', 'count'},
    'trend': {'time', 'series', 'temporal', 'seasonal', 'forecast',
              'over time', 'chronological', 'cyclical'},
    'time': {'trend', 'series', 'temporal', 'seasonal', 'date',
             'year', 'month', 'day', 'hour', 'minute', 'second',
             'week', 'quarter', 'chronological'},
    'compare': {'vs', 'versus', 'against', 'vs.', 'difference',
                'comparison', 'between'},
    'segment': {'cluster', 'group', 'partition', 'split',
                'divide', 'categorize'},
    'pattern': {'trend', 'shape', 'structure', 'regularity'},
}


def _expand(word):
    stemmed = _stem(word)
    result = {stemmed, word}
    if stemmed in _SYNONYMS:
        result.update(_SYNONYMS[stemmed])
    return result


# ── Stop words ─────────────────────────────────────────────────

_STOP = frozenset({
    'a', 'an', 'the', 'is', 'it', 'of', 'to', 'for', 'in',
    'on', 'and', 'or', 'with', 'at', 'by', 'from', 'as',
    'be', 'was', 'are', 'were', 'been', 'being', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'can', 'could', 'may', 'might', 'shall', 'should',
    'me', 'my', 'i', 'we', 'our', 'you', 'your', 'this',
    'that', 'these', 'those', 'some', 'any', 'all', 'no',
    'not', 'very', 'just', 'please', 'how', 'what', 'which',
    'who', 'when', 'where', 'why', 'about', 'than', 'then',
    'also', 'only', 'more', 'most', 'much', 'many',
})


def _tokenize(text):
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z_0-9]*', text.lower().strip())
    return [t for t in tokens if t not in _STOP and len(t) >= 2]


# ── Analysis type keyword signatures ───────────────────────────

_TYPE_KEYWORDS = {
    'distribution': {
        'distribution', 'histogram', 'kde', 'density', 'spread',
        'shape', 'range', 'skew', 'skewness', 'kurtosis',
        'quartile', 'percentile', 'central tendency', 'variance',
        'deviation', 'min', 'max', 'normal', 'bell',
    },
    'correlation': {
        'correlation', 'relationship', 'association', 'pairwise',
        'heatmap', 'multivariate', 'multicol', 'scatter',
        'crosstab', 'covariance', 'pearson', 'spearman',
        'dependence', 'interaction', 'between', 'vs',
    },
    'missing_values': {
        'missing', 'null', 'nan', 'empty', 'incomplete', 'blank',
        'gap', 'sparse', 'absent', 'void', 'na',
    },
    'categorical': {
        'categorical', 'category', 'bar', 'frequency', 'count',
        'enum', 'nominal', 'label', 'segment', 'discrete',
        'proportion', 'percentage', 'mode', 'contingency',
    },
    'outliers': {
        'outlier', 'anomaly', 'extreme', 'abnormal', 'unusual',
        'rare', 'suspicious', 'spike', 'deviation', 'aberrant',
        'exceptional', 'zscore', 'iqr', 'beyond',
    },
    'time_series': {
        'time', 'trend', 'seasonal', 'forecast', 'series',
        'date', 'temporal', 'chronological', 'cyclical',
        'over time', 'year', 'month', 'day', 'quarter',
    },
}


# ── TF weighting (log) ─────────────────────────────────────────

def _tf_weight(tokens):
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return {t: 1 + math.log10(c) if c > 0 else 0 for t, c in freq.items()}


# ── Main match function ───────────────────────────────────────

def match_query(query, columns=None):
    """Match a natural language query to an analysis type.

    Uses tokenization, stemming, synonym expansion, TF-weighted
    token overlap scoring, and optional column name extraction.

    Args:
        query: Free-form text query.
        columns: Optional list of column names for extraction.

    Returns:
        (analysis_type, confidence, mentioned_columns) tuple,
        or (None, 0.0, []) if no match.
    """
    raw = query.lower().strip()
    if not raw:
        return (None, 0.0, [])

    tokens = _tokenize(raw)
    if not tokens:
        return (None, 0.0, [])

    # Expand each token into its stem + synonyms
    expanded = {}
    for t in tokens:
        expanded[t] = _expand(t)

    # Compute TF weights for original tokens
    tf = _tf_weight(tokens)

    # Extract column name mentions before scoring
    mentioned_columns = _extract_column_mentions(tokens, columns)

    # Score each analysis type by weighted token overlap
    scores = {}
    for atype, keywords in _TYPE_KEYWORDS.items():
        score = 0.0
        total_weight = 0.0
        for t, syn_set in expanded.items():
            w = tf.get(t, 1.0)
            total_weight += w
            overlap = syn_set & keywords
            if overlap:
                score += w * (len(overlap) / max(len(keywords), 1) * 3)
            # bonus if token is a direct keyword match
            if t in keywords:
                score += w * 2.0
            # bonus for especially distinctive words
            if t in ('correlation', 'histogram', 'outlier',
                     'anomaly', 'distribution', 'categorical') and t in syn_set & keywords:
                score += w * 1.5
        scores[atype] = score / max(total_weight, 1)

    # Negation dampening — only dampen types mentioned near negation tokens
    negated = {'no', 'not', 'without', 'exclude', 'except', 'never'}
    neg_window = 3
    for i, t in enumerate(tokens):
        if t in negated:
            window = set(tokens[i+1:i+1+neg_window])
            for atype, keywords in _TYPE_KEYWORDS.items():
                if window & keywords:
                    scores[atype] *= 0.2

    # Best match
    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]

    if best_score < 0.1:
        return (None, 0.0, [])

    # Confidence scaling
    confidence = min(0.95, 0.5 + best_score * 1.5)
    if mentioned_columns:
        confidence = min(0.98, confidence + 0.08)

    return (best_type, round(confidence, 2), mentioned_columns)


def _extract_column_mentions(tokens, columns):
    """Extract column names from query tokens using fuzzy matching."""
    if not columns:
        return []
    mentioned = []
    for col in columns:
        col_key = col.lower().replace(' ', '_').replace('-', '_')
        for t in tokens:
            if t in col_key or col_key in t:
                mentioned.append(col)
                break
    return mentioned


def format_result(analysis_type, confidence, mentioned_columns=None):
    """Format match result as a display dictionary."""
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
        'columns': mentioned_columns or [],
    }
