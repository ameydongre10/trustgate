"""Issue profile feature definition and vectorization for damage prediction."""

from typing import Any, cast

import numpy as np

DAMAGE_FEATURE_NAMES = [
    "importance_weighted_null_rate",
    "max_null_rate",
    "mean_null_rate",
    "importance_weighted_psi",
    "max_psi",
    "mean_psi",
    "max_ks_stat",
    "outlier_rate",
    "duplicate_rate",
    "class_balance_ratio",
    "schema_columns_missing",
    "schema_columns_extra",
    "batch_size",
    "n_columns",
]

# Monotonic constraints: +1 means higher feature value cannot decrease predicted damage.
# 0 means unconstrained (e.g. batch_size, class_balance_ratio).
DAMAGE_MONOTONIC_CONSTRAINTS = [
    1,  # importance_weighted_null_rate
    1,  # max_null_rate
    1,  # mean_null_rate
    1,  # importance_weighted_psi
    1,  # max_psi
    1,  # mean_psi
    1,  # max_ks_stat
    1,  # outlier_rate
    1,  # duplicate_rate
    0,  # class_balance_ratio
    1,  # schema_columns_missing
    0,  # schema_columns_extra
    0,  # batch_size
    0,  # n_columns
]

FEATURE_HUMAN_LABELS: dict[str, str] = {
    "importance_weighted_null_rate": "Missing values in high-importance features",
    "max_null_rate": "Severe missingness in single column",
    "mean_null_rate": "Average missingness across columns",
    "importance_weighted_psi": "Population drift in critical features (PSI)",
    "max_psi": "Severe distribution shift in single column",
    "mean_psi": "Average distribution shift across batch",
    "max_ks_stat": "Kolmogorov-Smirnov distribution deviation",
    "outlier_rate": "Extreme outliers and sentinel values",
    "duplicate_rate": "Data duplication and potential leakage",
    "class_balance_ratio": "Severe class imbalance shift",
    "schema_columns_missing": "Missing expected schema columns",
    "schema_columns_extra": "Unexpected extra schema columns",
    "batch_size": "Batch sample size",
    "n_columns": "Number of columns",
}


def build_feature_vector(issue_metrics: dict[str, Any]) -> np.ndarray:
    """Extract ordered 1D numpy array of damage predictor features from an issue dictionary."""
    vec = []
    for name in DAMAGE_FEATURE_NAMES:
        val = issue_metrics.get(name, 0.0)
        try:
            val_f = float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            val_f = 0.0
        vec.append(val_f)
    return cast(np.ndarray, np.asarray(vec, dtype=np.float32))
