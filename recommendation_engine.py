import random
import logging

log = logging.getLogger('recommendation_engine')


REQUIRED_PROFILE_KEYS = [
    'numerical_cols', 'categorical_cols', 'missing_values',
    'missing_percentage', 'skewness', 'correlation_exists',
    'time_series_candidates', 'categorical_cardinality', 'has_outliers'
]

NUMERICAL_CATEGORIES = frozenset({'numerical', 'numerical_pairs'})


class RecommendationEngine:
    def __init__(self):
        self.analysis_catalog = {
            'distribution': {
                'applicable_to': 'numerical',
                'techniques': ['histogram', 'kde', 'boxplot'],
                'base_score': 0.8,
                'description': 'Analyze the distribution of values',
                'min_data_points': 5
            },
            'correlation': {
                'applicable_to': 'numerical_pairs',
                'techniques': ['heatmap', 'scatter', 'pairplot'],
                'base_score': 0.9,
                'description': 'Examine relationships between numerical features',
                'min_data_points': 10
            },
            'missing_values': {
                'applicable_to': 'any',
                'techniques': ['heatmap', 'bar'],
                'base_score': 0.7,
                'description': 'Analyze patterns in missing data',
                'min_data_points': 0
            },
            'categorical': {
                'applicable_to': 'categorical',
                'techniques': ['count', 'percentage', 'bar'],
                'base_score': 0.75,
                'description': 'Explore categorical feature distributions',
                'min_data_points': 3
            },
            'outliers': {
                'applicable_to': 'numerical',
                'techniques': ['boxplot', 'scatter'],
                'base_score': 0.6,
                'description': 'Identify and examine outliers',
                'min_data_points': 10
            },
            'time_series': {
                'applicable_to': 'datetime',
                'techniques': ['line', 'seasonal'],
                'base_score': 0.85,
                'description': 'Analyze time-based patterns and trends',
                'min_data_points': 10
            },
            'clustering': {
                'applicable_to': 'numerical',
                'techniques': ['scatter', 'cluster'],
                'base_score': 0.65,
                'description': 'Segment data into natural groups using k-means',
                'min_data_points': 10
            },
            'feature_importance': {
                'applicable_to': 'numerical_pairs',
                'techniques': ['heatmap', 'bar'],
                'base_score': 0.7,
                'description': 'Rank numerical features by mutual information',
                'min_data_points': 10
            }
        }
        
    def generate_recommendations(self, data_profile, user_preferences, quality_score=None, viewed_combos=None, interaction_history=None, col_affinity=None, epsilon=0.0):
        """Generate ranked recommendations based on data and user preferences."""
        missing_keys = [k for k in REQUIRED_PROFILE_KEYS if k not in data_profile]
        if missing_keys:
            raise ValueError(f"data_profile missing required keys: {missing_keys}")

        viewed_combos = viewed_combos or set()
        interaction_history = interaction_history or []
        col_affinity = col_affinity or {}
        recommendations = []

        for analysis_type, properties in self.analysis_catalog.items():
            if self._is_applicable(analysis_type, data_profile):
                base_score = properties['base_score']
                pref_score = user_preferences.get(analysis_type, 0.5)
                data_relevance = self._calculate_data_relevance(analysis_type, data_profile)
                
                final_score = base_score * pref_score * data_relevance
                
                if quality_score is not None:
                    quality_adjustment = 0.5 + 0.5 * quality_score
                    final_score = final_score * quality_adjustment
                else:
                    quality_adjustment = None
                
                applicable_columns = self._get_applicable_columns(analysis_type, data_profile)
                
                if applicable_columns:
                    data_factors = self._get_data_factors(analysis_type, data_profile)
                    sorted_cols = sorted(
                        applicable_columns,
                        key=lambda c: self._score_column_interestingness(analysis_type, c, data_profile),
                        reverse=True
                    )
                    recommendations.append({
                        'type': analysis_type,
                        'title': f"{analysis_type.title()} Analysis",
                        'description': properties['description'],
                        'techniques': properties['techniques'],
                        'score': final_score,
                        'base_score': base_score,
                        'pref_score': pref_score,
                        'data_relevance': data_relevance,
                        'quality_adjustment': quality_adjustment,
                        'diversity_penalty': None,
                        'data_factors': data_factors,
                        'columns': sorted_cols
                    })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        recommendations = self._bootstrap_scores(recommendations, data_profile, user_preferences)
        recommendations = self._apply_diversity_penalty(recommendations)
        recommendations = self._apply_novelty_penalty(recommendations, viewed_combos)
        recommendations = self._apply_avoidance_penalty(recommendations, interaction_history)
        recommendations = self._apply_affinity_boost(recommendations, col_affinity)
        recommendations = self._apply_epsilon_greedy(recommendations, epsilon=epsilon)
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)
    
    def _applicable_category(self, analysis_type):
        mapping = {
            'distribution': 'numerical',
            'correlation': 'numerical_pairs',
            'outliers': 'numerical',
            'categorical': 'categorical',
            'time_series': 'datetime',
            'missing_values': 'any',
            'clustering': 'numerical',
            'feature_importance': 'numerical_pairs',
        }
        return mapping.get(analysis_type, 'any')
    
    def _apply_diversity_penalty(self, recommendations):
        seen_categories = set()
        for rec in recommendations:
            cat = self._applicable_category(rec['type'])
            if cat in seen_categories and cat != 'any':
                rec['diversity_penalty'] = 0.85
                rec['score'] = rec['score'] * 0.85
            seen_categories.add(cat)
        return recommendations

    def _apply_novelty_penalty(self, recommendations, viewed_combos):
        if not viewed_combos:
            return recommendations
        for rec in recommendations:
            rec['novelty_penalty'] = None
            combo = (rec['type'], frozenset(rec.get('columns', [])))
            if combo in viewed_combos:
                rec['novelty_penalty'] = 0.6
                rec['score'] *= 0.6
            else:
                same_type_viewed = any(vc[0] == rec['type'] for vc in viewed_combos)
                if same_type_viewed:
                    rec['novelty_penalty'] = 0.85
                    rec['score'] *= 0.85
        return recommendations

    def _apply_avoidance_penalty(self, recommendations, interaction_history):
        if not interaction_history:
            return recommendations
        ignored_counts = {}
        for entry in interaction_history:
            if entry.get('action') == 'ignored':
                rtype = entry.get('recommendation_type')
                ignored_counts[rtype] = ignored_counts.get(rtype, 0) + 1
        for rec in recommendations:
            count = ignored_counts.get(rec['type'], 0)
            if count >= 3:
                rec['avoidance_penalty'] = 0.5
                rec['score'] *= 0.5
            elif count >= 1:
                rec['avoidance_penalty'] = max(0.6, 1.0 - 0.12 * count)
                rec['score'] *= rec['avoidance_penalty']
        return recommendations

    def _apply_affinity_boost(self, recommendations, col_affinity):
        if not col_affinity:
            return recommendations
        for rec in recommendations:
            cols = rec.get('columns', [])
            if not cols:
                continue
            scores = []
            for c in cols:
                scores.append(col_affinity.get(c, 0.0))
            mean_affinity = sum(scores) / len(scores) if scores else 0.0
            if mean_affinity > 0.0:
                boost = 1.0 + min(mean_affinity * 0.15, 0.10)
                rec['affinity_boost'] = boost
                rec['score'] *= boost
        return recommendations

    def _bootstrap_scores(self, recommendations, data_profile, user_preferences, n_iter=50):
        for rec in recommendations:
            scores = []
            atype = rec['type']
            cols = rec.get('columns', [])
            if not cols:
                rec['score_ci_lower'] = rec['score']
                rec['score_ci_upper'] = rec['score']
                continue
            for _ in range(n_iter):
                boot_cols = [random.choice(cols) for _ in range(len(cols))]
                boot_cols = list(set(boot_cols))
                if not boot_cols:
                    continue
                boot_profile = dict(data_profile)
                boot_profile[f'{atype}_cols_boot'] = boot_cols
                if atype == 'distribution':
                    boot_profile['numerical_cols'] = boot_cols
                elif atype == 'correlation':
                    boot_profile['numerical_cols'] = boot_cols
                elif atype == 'categorical':
                    boot_profile['categorical_cols'] = boot_cols
                elif atype == 'outliers':
                    boot_profile['has_outliers'] = {c: data_profile.get('has_outliers', {}).get(c, 0) for c in boot_cols}
                elif atype == 'clustering':
                    boot_profile['numerical_cols'] = boot_cols
                elif atype == 'feature_importance':
                    boot_profile['numerical_cols'] = boot_cols
                relevance = self._calculate_data_relevance(atype, boot_profile)
                boot_score = rec['base_score'] * user_preferences.get(atype, 0.5) * relevance
                if rec.get('quality_adjustment'):
                    boot_score *= rec['quality_adjustment']
                scores.append(boot_score)
            if scores:
                scores.sort()
                n = len(scores)
                rec['score_ci_lower'] = round(scores[max(0, int(n * 0.025))], 4)
                rec['score_ci_upper'] = round(scores[min(n - 1, int(n * 0.975))], 4)
            else:
                rec['score_ci_lower'] = rec['score']
                rec['score_ci_upper'] = rec['score']
        return recommendations

    def _apply_epsilon_greedy(self, recommendations, epsilon=0.1):
        if epsilon <= 0 or len(recommendations) <= 5:
            return recommendations
        if random.random() < epsilon:
            top_five = set(id(r) for r in recommendations[:5])
            candidates = [r for r in recommendations[5:] if id(r) not in top_five]
            if candidates:
                exploration_pick = random.choice(candidates)
                exploration_pick['exploration'] = True
                recommendations.remove(exploration_pick)
                recommendations.insert(random.randint(0, 4), exploration_pick)
                log.info(f"ε-greedy: swapped in {exploration_pick['type']} for exploration")
        return recommendations

    def _score_column_interestingness(self, analysis_type, column, data_profile):
        skewness = data_profile.get('skewness', {})
        has_outliers = data_profile.get('has_outliers', {})
        missing_pct = data_profile.get('missing_percentage', {})
        cardinality = data_profile.get('categorical_cardinality', {})
        
        if analysis_type in ('distribution', 'outliers', 'correlation'):
            skew = abs(skewness.get(column, 0) or 0)
            outlier_pct = has_outliers.get(column, 0)
            missing = missing_pct.get(column, 0)
            score = skew * 0.6 + outlier_pct * 0.4
            score = min(score, 1.0)
            if missing > 50:
                score *= 0.5
            return score
        
        if analysis_type == 'categorical':
            card = cardinality.get(column, 1)
            if card <= 1:
                return 0.0
            score = 1.0 - abs(card - 10) / 30
            score = max(0.3, min(0.9, score))
            return score
        
        if analysis_type == 'missing_values':
            return min(missing_pct.get(column, 0) / 100, 1.0)
        
        if analysis_type == 'time_series':
            return 1.0
        
        if analysis_type == 'clustering':
            skew = abs(skewness.get(column, 0) or 0)
            return 0.5 + 0.5 * min(skew, 1.0)
        
        if analysis_type == 'feature_importance':
            return 0.7
        
        return 0.5
    
    def _is_applicable(self, analysis_type, data_profile):
        if analysis_type == 'distribution':
            return len(data_profile['numerical_cols']) > 0
        
        elif analysis_type == 'correlation':
            return len(data_profile['numerical_cols']) > 1
        
        elif analysis_type == 'missing_values':
            has_missing = any(count > 0 for count in data_profile['missing_values'].values())
            return has_missing
        
        elif analysis_type == 'categorical':
            return len(data_profile['categorical_cols']) > 0
        
        elif analysis_type == 'outliers':
            return len(data_profile['has_outliers']) > 0
        
        elif analysis_type == 'time_series':
            return len(data_profile['time_series_candidates']) > 0
        
        elif analysis_type == 'clustering':
            return len(data_profile.get('numerical_cols', [])) >= 2
        
        elif analysis_type == 'feature_importance':
            return len(data_profile.get('numerical_cols', [])) >= 2
        
        return False
    
    def _calculate_data_relevance(self, analysis_type, data_profile):
        if analysis_type == 'distribution':
            skewed_cols = sum(1 for _, skew in data_profile['skewness'].items() 
                              if skew is not None and abs(skew) > 1)
            if skewed_cols > 0:
                return min(1.0, 0.7 + (0.3 * (skewed_cols / len(data_profile['numerical_cols']))))
            return 0.7
        
        elif analysis_type == 'correlation':
            mean_corr = data_profile.get('mean_pairwise_correlation', 0.0)
            return 0.5 + 0.5 * min(mean_corr, 1.0)
        
        elif analysis_type == 'missing_values':
            missing_percentages = [v for v in data_profile['missing_percentage'].values() if v > 0]
            if missing_percentages:
                avg_missing = sum(missing_percentages) / len(missing_percentages)
                return min(1.0, 0.5 + (0.5 * (avg_missing / 100)))
            return 0.5
        
        elif analysis_type == 'categorical':
            good_cardinality_cols = sum(1 for col, count in data_profile['categorical_cardinality'].items() 
                                      if 2 <= count <= 20)
            if good_cardinality_cols > 0:
                return min(1.0, 0.6 + (0.4 * (good_cardinality_cols / len(data_profile['categorical_cols']))))
            return 0.6
        
        elif analysis_type == 'outliers':
            outlier_percentages = list(data_profile['has_outliers'].values())
            if outlier_percentages:
                avg_outlier_pct = sum(outlier_percentages) / len(outlier_percentages)
                return min(1.0, 0.6 + (0.4 * min(avg_outlier_pct / 10, 1.0)))
            return 0.6
        
        elif analysis_type == 'time_series':
            num_candidates = len(data_profile['time_series_candidates'])
            return min(1.0, 0.7 + (0.3 * min(num_candidates / 3, 1.0)))
        
        elif analysis_type == 'clustering':
            num = len(data_profile.get('numerical_cols', []))
            return min(1.0, 0.5 + 0.5 * min(num / 5, 1.0))
        
        elif analysis_type == 'feature_importance':
            return 0.8
        
        return 0.5
    
    def _get_data_factors(self, analysis_type, data_profile):
        factors = []
        num = len(data_profile.get('numerical_cols', []))
        cat = len(data_profile.get('categorical_cols', []))
        if analysis_type == 'distribution':
            if num:
                factors.append(f"{num} numerical column{'s' if num != 1 else ''}")
            skewed = sum(1 for s in data_profile.get('skewness', {}).values() if s is not None and abs(s) > 1)
            if skewed:
                factors.append(f"{skewed} skewed distribution{'s' if skewed != 1 else ''}")
        elif analysis_type == 'correlation':
            if num >= 2:
                factors.append(f"{num} numerical columns available for pairwise analysis")
        elif analysis_type == 'missing_values':
            missing = [c for c, v in data_profile.get('missing_percentage', {}).items() if v > 0]
            if missing:
                factors.append(f"{len(missing)} column{'s' if len(missing) != 1 else ''} with missing data")
        elif analysis_type == 'categorical':
            if cat:
                factors.append(f"{cat} categorical column{'s' if cat != 1 else ''}")
            high_card = sum(1 for c in data_profile.get('categorical_cardinality', {}).values() if c > 20)
            if high_card:
                factors.append(f"{high_card} high-cardinality column{'s' if high_card != 1 else ''}")
        elif analysis_type == 'outliers':
            o_cols = len(data_profile.get('has_outliers', {}))
            if o_cols:
                factors.append(f"{o_cols} column{'s' if o_cols != 1 else ''} with outlier flags")
        elif analysis_type == 'time_series':
            ts = data_profile.get('time_series_candidates', [])
            if ts:
                factors.append(f"{len(ts)} time-related column{'s' if len(ts) != 1 else ''}")
        elif analysis_type == 'clustering':
            num = len(data_profile.get('numerical_cols', []))
            if num >= 2:
                factors.append(f"{num} numerical columns available for clustering")
        elif analysis_type == 'feature_importance':
            num = len(data_profile.get('numerical_cols', []))
            if num >= 2:
                factors.append(f"{num} numerical columns — computing pairwise mutual info")
        return factors

    def _get_applicable_columns(self, analysis_type, data_profile):
        if analysis_type == 'distribution':
            return data_profile['numerical_cols']
        
        elif analysis_type == 'correlation':
            return data_profile['numerical_cols']
        
        elif analysis_type == 'missing_values':
            return [col for col, count in data_profile['missing_values'].items() if count > 0]
        
        elif analysis_type == 'categorical':
            return data_profile['categorical_cols']
        
        elif analysis_type == 'outliers':
            return list(data_profile['has_outliers'].keys())
        
        elif analysis_type == 'time_series':
            return data_profile['time_series_candidates']
        
        elif analysis_type == 'clustering':
            return data_profile.get('numerical_cols', [])
        
        elif analysis_type == 'feature_importance':
            return data_profile.get('numerical_cols', [])
        
        return []