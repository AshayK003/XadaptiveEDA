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


def explain_recommendation(recommendation, data_profile, user_preferences, quality_report=None):
    """Explain why a recommendation scored the way it did, using actual values."""
    rec_type = recommendation['type']
    score = recommendation['score']
    pref_score = user_preferences.get(rec_type, 0.5)

    reasons = []

    base = recommendation.get('base_score')
    relevance = recommendation.get('data_relevance')
    quality_adj = recommendation.get('quality_adjustment')
    if base is not None and relevance is not None:
        parts = [f"{base:.2f} (base) × {pref_score:.1f} (priority) × {relevance:.2f} (data)"]
        if quality_adj is not None:
            score_no_q = base * pref_score * relevance
            parts.append(f"× {quality_adj:.2f} (quality)")
            reasons.append(f"Score = {' × '.join(parts)} = {score_no_q:.2f} × {quality_adj:.2f} = **{score:.2f}**")
            if quality_adj < 0.9:
                reasons.append("Data quality concerns reduced this score (see Data Quality Report).")
        else:
            reasons.append(f"Score = " + " x ".join(parts) + f" = **{score:.2f}**")
    else:
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
    technique_reasons = []
    for t in techniques[:2]:
        desc = TECHNIQUE_DESCRIPTIONS.get(t, t)
        if t == 'boxplot' and rec_type == 'outliers':
            outlier_cols = list(data_profile.get('has_outliers', {}).keys())[:3]
            if outlier_cols:
                desc += f" — best for visualizing distributions of flagged columns: {', '.join(outlier_cols)}"
        elif t == 'histogram' and rec_type == 'distribution':
            skewed = [c for c, s in data_profile.get('skewness', {}).items() if s is not None and abs(s) > 1]
            if skewed:
                desc += f" — reveals the skew direction in {', '.join(skewed[:2])}"
        technique_reasons.append(desc)

    if quality_report and recommendation.get('columns'):
        flagged = []
        for col in recommendation['columns']:
            if col in quality_report.sparse_columns:
                flagged.append(f"{col} is sparse")
            if col in quality_report.constant_columns:
                flagged.append(f"{col} is constant")
            if col in quality_report.mixed_type_columns:
                flagged.append(f"{col} has mixed types")
        if flagged:
            technique_reasons.append("[Quality note] " + "; ".join(flagged))

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


def compare_recommendations(rec1, rec2):
    """Generate side-by-side markdown comparison of two recommendations."""
    lines = []
    lines.append(f"| Component | **{rec1['title']}** | **{rec2['title']}** |")
    lines.append("|---|---|---|")
    def _fmt(val, fmt=".2f"):
        if val is None or isinstance(val, str):
            return 'N/A'
        return f"{val:{fmt}}"

    lines.append(f"| Base score | {_fmt(rec1.get('base_score'))} | {_fmt(rec2.get('base_score'))} |")
    lines.append(f"| Data relevance | {_fmt(rec1.get('data_relevance'))} | {_fmt(rec2.get('data_relevance'))} |")
    lines.append(f"| Your priority | {_fmt(rec1.get('pref_score'), '.1f')} | {_fmt(rec2.get('pref_score'), '.1f')} |")
    q1 = rec1.get('quality_adjustment')
    q2 = rec2.get('quality_adjustment')
    lines.append(f"| Quality adjustment | {f'{q1:.2f}' if q1 else 'N/A'} | {f'{q2:.2f}' if q2 else 'N/A'} |")
    d1 = rec1.get('diversity_penalty')
    d2 = rec2.get('diversity_penalty')
    lines.append(f"| Diversity penalty | {f'{d1:.2f}' if d1 else 'None'} | {f'{d2:.2f}' if d2 else 'None'} |")
    lines.append(f"| **Final score** | **{rec1.get('score', 0):.2f}** | **{rec2.get('score', 0):.2f}** |")
    f1 = rec1.get('data_factors', [])
    f2 = rec2.get('data_factors', [])
    lines.append(f"| Data factors | {'; '.join(f1) if f1 else 'None'} | {'; '.join(f2) if f2 else 'None'} |")
    return "\n".join(lines)


def global_explanation_summary(data_profile, quality_report, interaction_history, user_preferences):
    """Generate a session-wide markdown summary of exploration."""
    lines = []
    shape = data_profile.get('shape', (0, 0))
    qs = quality_report.overall_quality_score if quality_report else None

    lines.append("### Exploration Summary")
    lines.append("")
    lines.append(f"**Dataset:** {shape[0]:,} rows × {shape[1]} columns")
    if qs is not None:
        icon = "🟢" if qs >= 0.8 else ("🟡" if qs >= 0.5 else "🔴")
        lines.append(f"**Quality score:** {icon} {qs:.2f}")

    explored = set(e['recommendation_type'] for e in interaction_history if e.get('recommendation_type'))
    if explored:
        lines.append(f"\n**Explored analysis types:** {', '.join(sorted(explored))}")
    else:
        lines.append("\n**Explored analysis types:** No analysis types explored yet")

    feedback = [e for e in interaction_history if e.get('action') in ('liked', 'disliked')]
    if feedback:
        liked = sum(1 for e in feedback if e['action'] == 'liked')
        disliked = sum(1 for e in feedback if e['action'] == 'disliked')
        lines.append(f"**Feedback given:** 👍 {liked} liked, 👎 {disliked} disliked")
    else:
        lines.append("**Feedback given:** None yet")

    if user_preferences:
        diverged = {k: v for k, v in user_preferences.items() if abs(v - 0.5) > 0.1}
        if diverged:
            lines.append("\n**Adjusted preferences:**")
            for k, v in sorted(diverged.items(), key=lambda x: -abs(x[1] - 0.5)):
                direction = "↑" if v > 0.5 else "↓"
                lines.append(f"- {k}: {v:.2f} {direction}")
        else:
            lines.append("\n**Preferences:** All at default (0.5)")

    if quality_report:
        notes = []
        if quality_report.sparse_columns:
            notes.append(f"Sparse columns: {', '.join(quality_report.sparse_columns[:3])}")
        if quality_report.constant_columns:
            notes.append(f"Constant columns: {', '.join(quality_report.constant_columns[:3])}")
        if quality_report.mixed_type_columns:
            notes.append(f"Mixed types: {', '.join(quality_report.mixed_type_columns[:3])}")
        if quality_report.duplicate_rows:
            notes.append(f"Duplicate rows: {quality_report.duplicate_rows}")
        if notes:
            lines.append(f"\n**Data Quality Notes:** {'; '.join(notes)}")

    return "\n".join(lines)
