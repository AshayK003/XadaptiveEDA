import pandas as pd
import numpy as np
from constants import ANALYSIS_TYPES


REQUIRED_PROFILE_KEYS = [
    'numerical_cols', 'categorical_cols', 'missing_values',
    'missing_percentage', 'skewness', 'correlation_exists',
    'time_series_candidates', 'categorical_cardinality', 'has_outliers'
]


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
            }
        }
        
    def generate_recommendations(self, data_profile, user_preferences):
        """Generate ranked recommendations based on data and user preferences"""
        missing_keys = [k for k in REQUIRED_PROFILE_KEYS if k not in data_profile]
        if missing_keys:
            raise ValueError(f"data_profile missing required keys: {missing_keys}")

        recommendations = []

        for analysis_type, properties in self.analysis_catalog.items():
            if self._is_applicable(analysis_type, data_profile):
                # Calculate relevance score 
                base_score = properties['base_score']
                user_preference_score = user_preferences.get(analysis_type, 0.5)
                data_relevance = self._calculate_data_relevance(analysis_type, data_profile)
                
                final_score = base_score * user_preference_score * data_relevance
                
                # Get applicable columns for this analysis
                applicable_columns = self._get_applicable_columns(analysis_type, data_profile)
                
                if applicable_columns:
                    recommendations.append({
                        'type': analysis_type,
                        'title': f"{analysis_type.title()} Analysis",
                        'description': properties['description'],
                        'techniques': properties['techniques'],
                        'score': final_score,
                        'data_factors': self._get_data_factors(analysis_type, data_profile),
                        'columns': applicable_columns
                    })
        
        # Sort by final score
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)
    
    def _is_applicable(self, analysis_type, data_profile):
        """Check if analysis type is applicable to the current dataset"""
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
        
        return False
    
    def _calculate_data_relevance(self, analysis_type, data_profile):
        """Calculate how relevant an analysis is based on data characteristics"""
        if analysis_type == 'distribution':
            # More relevant for skewed distributions
            skewed_cols = sum(1 for _, skew in data_profile['skewness'].items() 
                              if skew is not None and abs(skew) > 1)
            if skewed_cols > 0:
                return min(1.0, 0.7 + (0.3 * (skewed_cols / len(data_profile['numerical_cols']))))
            return 0.7
        
        elif analysis_type == 'correlation':
            # Always highly relevant for multivariate numerical data
            return 1.0
        
        elif analysis_type == 'missing_values':
            # More relevant with higher percentage of missing values
            missing_percentages = [v for v in data_profile['missing_percentage'].values() if v > 0]
            if missing_percentages:
                avg_missing = sum(missing_percentages) / len(missing_percentages)
                return min(1.0, 0.5 + (0.5 * (avg_missing / 100)))
            return 0.5
        
        elif analysis_type == 'categorical':
            # More relevant for categorical columns with reasonable cardinality
            good_cardinality_cols = sum(1 for col, count in data_profile['categorical_cardinality'].items() 
                                      if 2 <= count <= 20)
            if good_cardinality_cols > 0:
                return min(1.0, 0.6 + (0.4 * (good_cardinality_cols / len(data_profile['categorical_cols']))))
            return 0.6
        
        elif analysis_type == 'outliers':
            # More relevant with higher percentage of outliers
            outlier_percentages = list(data_profile['has_outliers'].values())
            if outlier_percentages:
                avg_outlier_pct = sum(outlier_percentages) / len(outlier_percentages)
                return min(1.0, 0.6 + (0.4 * min(avg_outlier_pct / 10, 1.0)))
            return 0.6
        
        elif analysis_type == 'time_series':
            # More relevant with more time-series candidates
            num_candidates = len(data_profile['time_series_candidates'])
            return min(1.0, 0.7 + (0.3 * min(num_candidates / 3, 1.0)))
        
        return 0.5
    
    def _get_applicable_columns(self, analysis_type, data_profile):
        """Get columns applicable for a specific analysis type"""
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
        
        return []
    
    def _get_data_factors(self, analysis_type, data_profile):
        """Get data factors that influenced recommendation"""
        factors = []
        
        if analysis_type == 'distribution':
            # Check for skewness
            skewed_cols = [col for col, skew in data_profile['skewness'].items() 
                          if skew is not None and abs(skew) > 1]
            if skewed_cols:
                factors.append('skewed_distribution')
            
            # Check for multi-modal possibilities
            factors.append('distribution_analysis')
            
        elif analysis_type == 'correlation':
            factors.append('multivariate_numerical')
            
        elif analysis_type == 'missing_values':
            # Check for significant missing values
            high_missing = [col for col, pct in data_profile['missing_percentage'].items() 
                           if pct > 5]
            if high_missing:
                factors.append('high_missing_values')
            else:
                factors.append('some_missing_values')
                
        elif analysis_type == 'categorical':
            # Check for cardinality levels
            low_card = [col for col, count in data_profile['categorical_cardinality'].items() 
                       if 2 <= count <= 5]
            med_card = [col for col, count in data_profile['categorical_cardinality'].items() 
                       if 6 <= count <= 20]
            high_card = [col for col, count in data_profile['categorical_cardinality'].items() 
                        if count > 20]
            
            if low_card:
                factors.append('low_cardinality_categorical')
            if med_card:
                factors.append('medium_cardinality_categorical')
            if high_card:
                factors.append('high_cardinality_categorical')
                
        elif analysis_type == 'outliers':
            factors.append('has_outliers')
            
        elif analysis_type == 'time_series':
            factors.append('time_based_data')
            
        return factors 