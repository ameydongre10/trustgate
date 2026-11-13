"""Data quality metrics calculation: missingness, outliers, duplicates, and constants."""

import numpy as np
import polars as pl

from trustgate.ingest.schema import infer_series_type
from trustgate.profiling.models import ColumnProfile, QualitySummary

SENTINEL_NUMERIC_VALUES = {-999, -9999, 9999, 99999}
SENTINEL_STRING_VALUES = {"n/a", "na", "null", "none", "unknown", "?"}


def compute_column_quality(series: pl.Series, z_score_threshold: float = 3.5) -> ColumnProfile:
    """Compute quality, missingness, and outlier statistics for a single column."""
    col_name = series.name
    n_rows = len(series)
    inferred_type = infer_series_type(series)

    null_count = series.null_count()
    null_rate = null_count / n_rows if n_rows > 0 else 0.0
    distinct_count = series.n_unique()

    min_val = None
    max_val = None
    mean_val = None
    std_val = None
    zero_variance = False
    outlier_count = 0
    outlier_rate = 0.0

    if inferred_type == "numeric":
        non_null = series.drop_nulls()
        if len(non_null) > 0 and non_null.dtype.is_numeric():
            arr = non_null.to_numpy()
            clean_arr = arr[np.isfinite(arr)]
            if len(clean_arr) > 0:
                min_val = float(np.min(clean_arr))
                max_val = float(np.max(clean_arr))
                mean_val = float(np.mean(clean_arr))
                std_val = float(np.std(clean_arr))

                # Check zero-variance
                if std_val == 0.0:
                    zero_variance = True
                else:
                    # Z-score outlier detection with zero-variance guard
                    z_scores = np.abs((clean_arr - mean_val) / std_val)
                    z_outliers: int = int(np.sum(z_scores > z_score_threshold))

                    # Sentinel values detection
                    sentinel_count = sum(1 for v in clean_arr if v in SENTINEL_NUMERIC_VALUES)
                    outlier_count = int(z_outliers + sentinel_count)
                    outlier_rate = min(1.0, outlier_count / n_rows) if n_rows > 0 else 0.0

    elif inferred_type in ("categorical", "text"):
        # Check string sentinels
        non_null_str = series.drop_nulls().cast(pl.Utf8)
        if len(non_null_str) > 0:
            sentinels = non_null_str.filter(
                non_null_str.str.to_lowercase().is_in(SENTINEL_STRING_VALUES)
            )
            outlier_count = len(sentinels)
            outlier_rate = outlier_count / n_rows if n_rows > 0 else 0.0

    return ColumnProfile(
        column_name=col_name,
        inferred_type=inferred_type,
        null_count=null_count,
        null_rate=null_rate,
        distinct_count=distinct_count,
        min_value=min_val,
        max_value=max_val,
        mean=mean_val,
        std=std_val,
        zero_variance=zero_variance,
        outlier_count=outlier_count,
        outlier_rate=outlier_rate,
    )


def compute_batch_quality_summary(
    df: pl.DataFrame,
    column_profiles: dict[str, ColumnProfile],
    high_missingness_threshold: float = 0.20,
) -> QualitySummary:
    """Aggregate overall batch quality summary across all columns."""
    total_rows = len(df)
    total_columns = len(df.columns)

    if total_rows == 0 or total_columns == 0:
        return QualitySummary(
            total_rows=total_rows,
            total_columns=total_columns,
            overall_null_rate=0.0,
            duplicate_row_count=0,
            duplicate_rate=0.0,
            constant_columns=[],
            high_missingness_columns=[],
        )

    # Compute overall null rate
    total_cells = total_rows * total_columns
    total_nulls = sum(prof.null_count for prof in column_profiles.values())
    overall_null_rate = total_nulls / total_cells if total_cells > 0 else 0.0

    # Compute duplicate row count
    dup_mask = df.is_duplicated()
    dup_count = int(dup_mask.sum())
    dup_rate = dup_count / total_rows if total_rows > 0 else 0.0

    constant_cols = [
        name
        for name, prof in column_profiles.items()
        if prof.zero_variance or prof.distinct_count <= 1
    ]
    high_missing_cols = [
        name
        for name, prof in column_profiles.items()
        if prof.null_rate >= high_missingness_threshold
    ]

    return QualitySummary(
        total_rows=total_rows,
        total_columns=total_columns,
        overall_null_rate=overall_null_rate,
        duplicate_row_count=dup_count,
        duplicate_rate=dup_rate,
        constant_columns=constant_cols,
        high_missingness_columns=high_missing_cols,
    )
