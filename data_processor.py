import pandas as pd
import os
import warnings
from data_quality import cleanse as _cleanse, QualityReport


class DataProcessor:
    def load_data(self, file_or_path):
        """Load data from various sources and return DataFrame"""
        if isinstance(file_or_path, str):
            if os.path.getsize(file_or_path) == 0:
                raise ValueError("File is empty")
            file_extension = os.path.splitext(file_or_path)[1].lower()
            if file_extension == '.csv':
                return pd.read_csv(file_or_path)
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(file_or_path)
            elif file_extension == '.json':
                return pd.read_json(file_or_path)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")

        file_name = getattr(file_or_path, 'name', '')
        file_extension = os.path.splitext(file_name)[1].lower()

        if file_extension == '.csv':
            return pd.read_csv(file_or_path)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_or_path)
        elif file_extension == '.json':
            return pd.read_json(file_or_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def cleanse(self, df: pd.DataFrame, skip_name_normalization: bool = False) -> tuple[pd.DataFrame, QualityReport]:
        """Normalize, validate, and report on data quality. Returns (cleaned_df, report)."""
        return _cleanse(df, skip_name_normalization=skip_name_normalization)

    def profile_dataset(self, df):
        """Extract key dataset characteristics"""
        if df.empty:
            return {
                'shape': (0, 0),
                'dtypes': {},
                'missing_values': {},
                'missing_percentage': {},
                'numerical_cols': [],
                'categorical_cols': [],
                'unique_counts': {},
                'skewness': {},
                'correlation_exists': False,
                'time_series_candidates': [],
                'categorical_cardinality': {},
                'has_outliers': {}
            }

        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Calculate skewness for numerical columns
        skewness = {}
        for col in numerical_cols:
            if not df[col].isnull().all():  # Skip if all values are null
                try:
                    skewness[col] = df[col].skew()
                except Exception:
                    skewness[col] = None
        
        # Get unique counts for all columns
        unique_counts = {}
        for col in df.columns:
            unique_counts[col] = df[col].nunique()
        
        # Check for potential datetime columns
        time_series_candidates = []
        # Common datetime formats to check
        date_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y', 
            '%Y/%m/%d', '%b %d %Y', '%B %d %Y', '%d %b %Y',
            '%d %B %Y', '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S'
        ]
        
        # First check column names
        for col in df.columns:
            col_lower = col.lower()
            if ('date' in col_lower or 'time' in col_lower or 
                'day' in col_lower or 'year' in col_lower or 
                'month' in col_lower):
                time_series_candidates.append(col)
        
        # Then check object columns for convertible date formats
        for col in df.select_dtypes(include=['object']).columns:
            if col in time_series_candidates:  # Skip already identified columns
                continue
            
            sample_vals = df[col].dropna()
            if sample_vals.empty:
                continue
            
            sample_val = str(sample_vals.iloc[0])
            # Quick heuristic: must have digits, be at least 6 chars, and contain date-like separators
            if not any(c.isdigit() for c in sample_val) or len(sample_val) < 6:
                continue
            
            date_detected = False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for date_format in date_formats:
                    try:
                        pd.to_datetime(sample_val, format=date_format)
                        date_detected = True
                        break
                    except Exception:
                        continue
            
            if date_detected or (any(x in sample_val.lower() for x in ['-', '/', ':']) and len(sample_val) >= 8):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        null_pct = df[col].isnull().mean()
                        convert_attempt = pd.to_datetime(df[col], errors='coerce', dayfirst=True, cache=True)
                        new_null_pct = convert_attempt.isnull().mean()
                        if new_null_pct - null_pct < 0.2:
                            time_series_candidates.append(col)
                except Exception:
                    pass
        
        # Return profile dictionary
        return {
            'shape': df.shape,
            'dtypes': dict(df.dtypes.astype(str)),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols,
            'unique_counts': unique_counts,
            'skewness': skewness,
            'correlation_exists': len(numerical_cols) > 1,
            'time_series_candidates': time_series_candidates,
            'categorical_cardinality': {col: df[col].nunique() for col in categorical_cols},
            'has_outliers': self._check_outliers(df, numerical_cols)
        }
    
    def _check_outliers(self, df, numerical_cols):
        """Check for potential outliers in numerical columns"""
        outlier_cols = {}
        
        for col in numerical_cols:
            if not df[col].isnull().all():
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                lower_bound = q1 - 1.5 * iqr
                
                outliers = df[(df[col] > upper_bound) | (df[col] < lower_bound)]
                outlier_percentage = len(outliers) / len(df) * 100
                
                if outlier_percentage > 0:
                    outlier_cols[col] = outlier_percentage 
        
        return outlier_cols  # Return the outlier columns dictionary 