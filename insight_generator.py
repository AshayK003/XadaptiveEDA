import pandas as pd


TECHNIQUE_DESCRIPTIONS = {
    'histogram': "Histograms show the frequency distribution of values across bins",
    'kde': "Kernel Density Estimation plots the smoothed probability density of the data",
    'boxplot': "Boxplots display quartiles, median, and values beyond 1.5x IQR",
    'heatmap': "Heatmaps use color intensity to show the strength of relationships between variables",
    'scatter': "Scatter plots show the relationship between two variables as points",
    'pairplot': "Pair plots create scatter matrices for all numerical variable combinations",
    'bar': "Bar charts compare values across categories using bar heights",
    'count': "Count plots show the frequency of each distinct value",
    'percentage': "Percentage charts show the relative proportion of each category",
    'line': "Line charts display trends by connecting data points in order",
    'seasonal': "Seasonal decomposition separates trend, seasonal patterns, and residuals"
}


def generate_insights(data_profile):
    """Generate factual observations from the actual data profile values."""
    insights = []

    shape = data_profile.get('shape', (0, 0))
    insights.append(f"Dataset has {shape[0]:,} rows and {shape[1]} columns.")

    numerical_cols = data_profile.get('numerical_cols', [])
    if numerical_cols:
        insights.append(f"Contains {len(numerical_cols)} numerical column(s): {', '.join(numerical_cols)}.")

    categorical_cols = data_profile.get('categorical_cols', [])
    if categorical_cols:
        insights.append(f"Contains {len(categorical_cols)} categorical column(s): {', '.join(categorical_cols)}.")

    missing = data_profile.get('missing_percentage', {})
    cols_with_missing = {col: pct for col, pct in missing.items() if pct > 0}
    if cols_with_missing:
        worst = max(cols_with_missing, key=cols_with_missing.get)
        insights.append(f"Missing values detected — '{worst}' has the most ({cols_with_missing[worst]:.1f}%).")
    else:
        insights.append("No missing values found.")

    outliers = data_profile.get('has_outliers', {})
    if outliers:
        worst_out = max(outliers, key=outliers.get)
        insights.append(f"Potential outliers found — '{worst_out}' has {outliers[worst_out]:.1f}% values beyond 1.5x IQR.")

    skewness = data_profile.get('skewness', {})
    skewed = {col: s for col, s in skewness.items() if s is not None and abs(s) > 1}
    if skewed:
        worst_skew = max(skewed, key=lambda c: abs(skewed[c]))
        direction = "right" if skewed[worst_skew] > 0 else "left"
        insights.append(f"'{worst_skew}' is {direction}-skewed (skewness={skewed[worst_skew]:.2f}).")

    time_cols = data_profile.get('time_series_candidates', [])
    if time_cols:
        insights.append(f"Potential time-based columns: {', '.join(time_cols)}.")

    return insights


def explain_recommendation(recommendation, data_profile, user_preferences):
    """Explain why a recommendation scored the way it did, using actual values."""
    rec_type = recommendation['type']
    score = recommendation['score']
    pref_score = user_preferences.get(rec_type, 0.5)

    reasons = []

    if score >= 0.5:
        reasons.append(f"Relevance score: {score:.2f} (scale 0-1).")

    skewed_cols = {col: s for col, s in data_profile.get('skewness', {}).items()
                   if s is not None and abs(s) > 1}
    if rec_type == 'distribution' and skewed_cols:
        worst = max(skewed_cols, key=lambda c: abs(skewed_cols[c]))
        d = "right" if skewed_cols[worst] > 0 else "left"
        reasons.append(f"'{worst}' is {d}-skewed — distribution analysis helps characterize this.")

    if rec_type == 'correlation':
        num_cols = data_profile.get('numerical_cols', [])
        if len(num_cols) > 1:
            reasons.append(f"{len(num_cols)} numerical columns available for correlation analysis.")

    if rec_type == 'missing_values':
        missing = {c: p for c, p in data_profile.get('missing_percentage', {}).items() if p > 0}
        if missing:
            top = max(missing, key=missing.get)
            reasons.append(f"'{top}' is {missing[top]:.1f}% missing — may need investigation.")

    if rec_type == 'outliers':
        outliers = data_profile.get('has_outliers', {})
        if outliers:
            top = max(outliers, key=outliers.get)
            reasons.append(f"'{top}' has {outliers[top]:.1f}% potential outliers.")

    if rec_type == 'time_series':
        time_cols = data_profile.get('time_series_candidates', [])
        if time_cols:
            reasons.append(f"{len(time_cols)} time-related column(s) detected.")

    if pref_score > 0.6:
        reasons.append(f"Your priority for this type is {pref_score:.1f} — higher than default.")
    elif pref_score < 0.4:
        reasons.append(f"Your priority for this type is {pref_score:.1f} — lower than default.")
    else:
        reasons.append("Priority is at the default level (0.5).")

    techniques = recommendation.get('techniques', [])
    technique_reasons = [TECHNIQUE_DESCRIPTIONS.get(t, t) for t in techniques[:2]]

    return {
        'reasons': reasons,
        'technique_reasons': technique_reasons
    }


def explain_user_preferences(preferences):
    """Describe current user priority settings in plain text."""
    if not preferences:
        return "All priorities are at their default values (0.5)."

    sorted_prefs = sorted(preferences.items(), key=lambda x: x[1], reverse=True)
    top = sorted_prefs[0]
    bottom = sorted_prefs[-1]

    parts = []
    if top[1] > 0.6:
        parts.append(f"Highest priority: {top[0].replace('_', ' ')} ({top[1]:.1f})")
    if bottom[1] < 0.4:
        parts.append(f"Lowest priority: {bottom[0].replace('_', ' ')} ({bottom[1]:.1f})")

    if not parts:
        return "All priorities are near the default level (0.5)."

    return " | ".join(parts)
