"""
Data quality and preprocessing pipeline.

Inserts between load_data() and profile_dataset() to normalize,
validate, and report on data quality without changing existing interfaces.
"""

import pandas as pd
import numpy as np
import re
from dataclasses import dataclass, field
from typing import Any


# ── Configuration ─────────────────────────────────────────────

MISSING_TOKENS = frozenset({
    "", " ", "NA", "N/A", "N / A", "NULL", "null", "None", "none",
    "-", "--", "?", "#N/A", "#N/A N/A", "nan", "NaN", "NaT", "n/a"
})

SPARSE_THRESHOLD = 0.5           # columns exceeding this missing ratio qualify
LARGE_DATASET_ROWS = 100_000     # sampling threshold
TYPE_INFERENCE_SAMPLE = 100      # rows to sample for type inference
NUMERIC_CONFIDENCE = 0.9         # min fraction of parsable values to cast


# ── Quality Report ────────────────────────────────────────────

@dataclass
class QualityReport:
    completeness: float = 1.0
    uniqueness: float = 1.0
    datatype_consistency: dict[str, dict] = field(default_factory=dict)
    duplicate_rows: int = 0
    null_percentages: dict[str, float] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    overall_quality_score: float = 1.0
    warnings: list[str] = field(default_factory=list)
    num_cols_after_cleanse: int = 0
    row_count_after_cleanse: int = 0
    rows_removed_fully_empty: int = 0
    cols_removed_fully_empty: int = 0
    sparse_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    mixed_type_columns: list[str] = field(default_factory=list)
    duplicate_columns_renamed: list[tuple[str, str]] = field(default_factory=list)


# ── Pipeline ──────────────────────────────────────────────────

class DataQualityPipeline:
    """Orchestrates data quality checks and preprocessing steps."""

    def __init__(self):
        self.warnings: list[str] = []

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
        """Run full pipeline. Returns (cleaned_df, QualityReport)."""
        if df.empty:
            report = QualityReport(warnings=["Empty dataset — no processing applied"])
            return df, report

        df = df.copy()  # never mutate upstream

        df = self._normalize_missing(df)

        # Dedup and normalize column names BEFORE column-level operations
        df, renamed = self._deduplicate_columns(df)
        if renamed:
            self.warnings.append(f"Renamed {len(renamed)} duplicate column name(s)")

        df = self._normalize_column_names(df)

        df = self._normalize_infinite(df)

        df, empty_cols = self._remove_empty_columns(df)
        cols_removed = len(empty_cols)
        if empty_cols:
            self.warnings.append(f"Removed {len(empty_cols)} fully empty column(s)")

        n_before = len(df)
        df = self._remove_empty_rows(df)
        rows_removed = n_before - len(df)

        sparse_cols = self._detect_sparse_columns(df)
        constant_cols = self._detect_constant_columns(df)
        mixed_cols = self._detect_mixed_types(df)

        df = self._infer_types(df)

        duplicate_rows = int(df.duplicated().sum())

        report = QualityReport(
            duplicate_rows=duplicate_rows,
            duplicate_columns_renamed=renamed,
            sparse_columns=sparse_cols,
            constant_columns=constant_cols,
            mixed_type_columns=mixed_cols,
            rows_removed_fully_empty=rows_removed,
            cols_removed_fully_empty=cols_removed,
            num_cols_after_cleanse=df.shape[1],
            row_count_after_cleanse=df.shape[0],
        )

        self._compute_quality_metrics(df, report)
        self._collect_warnings(report)
        report.warnings = self.warnings

        return df, report

    # ── Step implementations ──────────────────────────────────

    def _normalize_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace common missing-value tokens with np.nan."""
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].apply(
                    lambda x: np.nan if isinstance(x, str) and x.strip() in MISSING_TOKENS else x
                )
        return df

    def _normalize_infinite(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace inf/-inf with NaN."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return df

    def _remove_empty_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        empty = [col for col in df.columns if df[col].isnull().all()]
        if empty:
            df = df.drop(columns=empty)
        return df, empty

    def _remove_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(how='all').reset_index(drop=True)

    def _deduplicate_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
        seen: dict[str, int] = {}
        renamed: list[tuple[str, str]] = []
        new_columns: list[str] = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_name = f"{col}_{seen[col]}"
                new_columns.append(new_name)
                renamed.append((col, new_name))
            else:
                seen[col] = 0
                new_columns.append(str(col))
        df.columns = new_columns
        return df, renamed

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols: list[str] = []
        for col in df.columns:
            new = str(col).strip().lower().replace(" ", "_")
            new = re.sub(r'[^\w]', '_', new)
            new = re.sub(r'_+', '_', new).strip('_')
            new_cols.append(new if new else "column")
        df.columns = new_cols
        return df

    def _detect_sparse_columns(self, df: pd.DataFrame) -> list[str]:
        return [col for col in df.columns if df[col].isnull().mean() > SPARSE_THRESHOLD]

    def _detect_constant_columns(self, df: pd.DataFrame) -> list[str]:
        return [col for col in df.columns if df[col].nunique(dropna=False) <= 1]

    def _detect_mixed_types(self, df: pd.DataFrame) -> list[str]:
        mixed: list[str] = []
        for col in df.select_dtypes(include=['object']).columns:
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            types = set(type(v).__name__ for v in non_null.head(TYPE_INFERENCE_SAMPLE))
            if len(types) > 1:
                mixed.append(col)
        return mixed

    def _infer_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Safely infer numeric and datetime with high confidence only."""
        for col in df.columns:
            if df[col].isnull().mean() > 0.8 or df[col].nunique(dropna=False) <= 1:
                continue

            if not pd.api.types.is_object_dtype(df[col]):
                continue

            non_null = df[col].dropna()
            if non_null.empty:
                continue

            sample = non_null.head(TYPE_INFERENCE_SAMPLE)

            # Numeric check
            numeric_test = pd.to_numeric(sample, errors='coerce')
            if numeric_test.notna().mean() > NUMERIC_CONFIDENCE:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                continue

            # Datetime — only if column name suggests it
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['date', 'time', 'timestamp', 'datetime', 'created', 'updated']):
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if parsed.notna().mean() > 0.8:
                        df[col] = parsed
                except Exception:
                    pass

        return df

    def _compute_quality_metrics(self, df: pd.DataFrame, report: QualityReport) -> None:
        total_cells = df.shape[0] * df.shape[1]
        report.completeness = 1.0 - (df.isnull().sum().sum() / total_cells) if total_cells else 1.0

        report.null_percentages = (df.isnull().sum() / len(df) * 100).to_dict() if len(df) else {}

        uniqueness_ratios = []
        for col in df.columns:
            n = df[col].nunique()
            uniqueness_ratios.append(n / len(df) if len(df) else 1.0)
        report.uniqueness = float(np.mean(uniqueness_ratios)) if uniqueness_ratios else 1.0

        report.memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        report.datatype_consistency = {}
        for col in df.columns:
            if col in report.mixed_type_columns:
                report.datatype_consistency[col] = {"consistent": False, "note": "Mixed data types detected"}
            else:
                report.datatype_consistency[col] = {"consistent": True, "note": ""}

        weights = {
            'completeness': 0.3, 'uniqueness': 0.2,
            'no_duplicates': 0.15, 'no_sparse': 0.15,
            'no_constant': 0.1, 'no_mixed': 0.1
        }
        n = len(df)
        dupe_score = 1.0 - min(report.duplicate_rows / max(n, 1), 0.5) if n else 1.0
        nc = report.num_cols_after_cleanse or 1
        sparse_score = 1.0 - min(len(report.sparse_columns) / max(nc, 1), 0.5)
        constant_score = 1.0 - min(len(report.constant_columns) / max(nc, 1), 0.5)
        mixed_score = 1.0 - min(len(report.mixed_type_columns) / max(nc, 1), 0.5)

        report.overall_quality_score = (
            weights['completeness'] * report.completeness +
            weights['uniqueness'] * report.uniqueness +
            weights['no_duplicates'] * dupe_score +
            weights['no_sparse'] * sparse_score +
            weights['no_constant'] * constant_score +
            weights['no_mixed'] * mixed_score
        )

    def _collect_warnings(self, report: QualityReport) -> None:
        if report.sparse_columns:
            for col in report.sparse_columns:
                pct = report.null_percentages.get(col, 0)
                self.warnings.append(f"'{col}' is sparse ({pct:.0f}% missing)")
        if report.constant_columns:
            self.warnings.append(
                f"{len(report.constant_columns)} constant column(s): {', '.join(report.constant_columns[:5])}"
            )
        if report.mixed_type_columns:
            for col in report.mixed_type_columns:
                self.warnings.append(f"'{col}' has mixed data types")
        if report.duplicate_rows > 0:
            self.warnings.append(f"Found {report.duplicate_rows} duplicate row(s)")
        if report.memory_usage_mb > 100:
            self.warnings.append(f"Large dataset ({report.memory_usage_mb:.0f} MB)")


# ── Convenience function ──────────────────────────────────────

def cleanse(df: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    """One-shot entry point for the full pipeline."""
    return DataQualityPipeline().run(df)
