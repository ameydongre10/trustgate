"""Parameterized corruption injectors for quality, drift, label noise, and schema anomalies."""

from typing import Any

import numpy as np
import polars as pl

from trustgate.utils.seeding import get_rng


def inject_mcar(
    df: pl.DataFrame,
    columns: list[str],
    severity: float,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Missing Completely At Random (MCAR) injection."""
    rng = get_rng(seed)
    result = df.clone()
    target_cols = [c for c in columns if c in result.columns]
    affected_cells = 0

    for col in target_cols:
        mask = rng.uniform(0, 1, size=result.height) < severity
        affected_cells += int(np.sum(mask))
        mask_series = pl.Series(mask)
        result = result.with_columns(
            pl.when(mask_series).then(None).otherwise(pl.col(col)).alias(col)
        )

    return result, {
        "injector": "mcar",
        "severity": severity,
        "columns": target_cols,
        "affected_cells": affected_cells,
    }


def inject_mar(
    df: pl.DataFrame,
    target_col: str,
    condition_col: str,
    severity: float,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Missing At Random (MAR): missingness in target_col conditioned on condition_col values."""
    if target_col not in df.columns or condition_col not in df.columns:
        return df, {"injector": "mar", "skipped": True}

    rng = get_rng(seed)
    result = df.clone()
    cond_vals = result[condition_col].drop_nulls().to_numpy()
    if len(cond_vals) == 0:
        return df, {"injector": "mar", "skipped": True}

    median_val = np.median(cond_vals)
    base_cond = result[condition_col].to_numpy()
    probs = np.where(base_cond > median_val, min(1.0, severity * 2.0), severity * 0.2)
    mask = rng.uniform(0, 1, size=len(probs)) < probs

    mask_series = pl.Series(mask)
    result = result.with_columns(
        pl.when(mask_series).then(None).otherwise(pl.col(target_col)).alias(target_col)
    )

    return result, {
        "injector": "mar",
        "severity": severity,
        "target_col": target_col,
        "condition_col": condition_col,
        "affected_cells": int(np.sum(mask)),
    }


def inject_mnar(
    df: pl.DataFrame,
    target_col: str,
    severity: float,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Missing Not At Random (MNAR): missingness probability depends on column's own magnitude."""
    if target_col not in df.columns:
        return df, {"injector": "mnar", "skipped": True}

    rng = get_rng(seed)
    result = df.clone()
    vals = result[target_col].to_numpy()
    if not np.issubdtype(vals.dtype, np.number):
        return df, {"injector": "mnar", "skipped": True}

    min_v: float = float(np.nanmin(vals))
    max_v: float = float(np.nanmax(vals))
    span = max_v - min_v if max_v > min_v else 1.0
    normalized = np.clip((vals - min_v) / span, 0.0, 1.0)
    probs = np.clip(normalized * (severity * 2.5), 0.0, 0.95)
    mask = rng.uniform(0, 1, size=len(probs)) < probs

    mask_series = pl.Series(mask)
    result = result.with_columns(
        pl.when(mask_series).then(None).otherwise(pl.col(target_col)).alias(target_col)
    )

    return result, {
        "injector": "mnar",
        "severity": severity,
        "target_col": target_col,
        "affected_cells": int(np.sum(mask)),
    }


def inject_mean_shift(
    df: pl.DataFrame,
    columns: list[str],
    shift_stds: float,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Covariate drift: shift numeric column mean by k * std."""
    result = df.clone()
    target_cols = [c for c in columns if c in result.columns]

    for col in target_cols:
        series = result[col].to_numpy()
        if np.issubdtype(series.dtype, np.number):
            std_val = float(np.nanstd(series))
            shift_amount = shift_stds * (std_val if std_val > 0 else 1.0)
            shifted = series + shift_amount
            result = result.with_columns(pl.Series(name=col, values=shifted))

    return result, {
        "injector": "mean_shift",
        "shift_stds": shift_stds,
        "columns": target_cols,
    }


def inject_variance_scale(
    df: pl.DataFrame,
    columns: list[str],
    scale_factor: float,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Covariate drift: scale column variance by multiplying distance from mean."""
    result = df.clone()
    target_cols = [c for c in columns if c in result.columns]

    for col in target_cols:
        series = result[col].to_numpy()
        if np.issubdtype(series.dtype, np.number):
            mean_val = float(np.nanmean(series))
            scaled = mean_val + (series - mean_val) * scale_factor
            result = result.with_columns(pl.Series(name=col, values=scaled))

    return result, {
        "injector": "variance_scale",
        "scale_factor": scale_factor,
        "columns": target_cols,
    }


def inject_unit_error(
    df: pl.DataFrame,
    column: str,
    multiplier: float,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Covariate drift: scale column by factor (e.g. 100.0 or 0.01) simulating unit mismatches."""
    if column not in df.columns:
        return df, {"injector": "unit_error", "skipped": True}

    result = df.clone()
    series = result[column].to_numpy()
    if np.issubdtype(series.dtype, np.number):
        multiplied = series * multiplier
        result = result.with_columns(pl.Series(name=column, values=multiplied))

    return result, {
        "injector": "unit_error",
        "column": column,
        "multiplier": multiplier,
    }


def inject_unseen_categories(
    df: pl.DataFrame,
    column: str,
    severity: float,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Covariate drift: introduce out-of-vocabulary category tokens."""
    if column not in df.columns:
        return df, {"injector": "unseen_categories", "skipped": True}

    rng = get_rng(seed)
    result = df.clone()
    series = result[column].to_list()
    mask = rng.uniform(0, 1, size=len(series)) < severity
    novel_tokens = ["UNKNOWN_VARIANT_X", "NOVEL_OOD_CATEGORY", "LEGACY_CLASS_Z"]

    new_series = [str(rng.choice(novel_tokens)) if m else s for s, m in zip(series, mask, strict=False)]
    result = result.with_columns(pl.Series(name=column, values=new_series))

    return result, {
        "injector": "unseen_categories",
        "column": column,
        "severity": severity,
        "affected_cells": int(np.sum(mask)),
    }


def inject_concept_drift(
    df: pl.DataFrame,
    target_col: str,
    condition_col: str,
    severity: float,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Concept drift: alter P(y|x) conditional on feature values."""
    if target_col not in df.columns or condition_col not in df.columns:
        return df, {"injector": "concept_drift", "skipped": True}

    rng = get_rng(seed)
    result = df.clone()
    y = result[target_col].to_numpy()
    cond = result[condition_col].to_numpy()

    if np.issubdtype(cond.dtype, np.number):
        med = np.nanmedian(cond)
        condition_mask = cond > med
    else:
        condition_mask = np.ones(len(cond), dtype=bool)

    flip_prob = np.where(condition_mask, severity * 2.0, severity * 0.1)
    flip_mask = rng.uniform(0, 1, size=len(y)) < flip_prob

    if np.issubdtype(y.dtype, np.integer) and set(np.unique(y)).issubset({0, 1}):
        # Binary classification flip
        new_y = np.where(flip_mask, 1 - y, y)
    else:
        # Continuous regression shift
        std_y = float(np.nanstd(y)) if len(y) > 1 else 1.0
        new_y = np.where(flip_mask, y + 2.0 * std_y, y)

    result = result.with_columns(pl.Series(name=target_col, values=new_y))

    return result, {
        "injector": "concept_drift",
        "target_col": target_col,
        "condition_col": condition_col,
        "severity": severity,
        "flipped_labels": int(np.sum(flip_mask)),
    }


def inject_label_noise(
    df: pl.DataFrame,
    target_col: str,
    noise_type: str = "symmetric",
    severity: float = 0.10,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Inject symmetric or asymmetric label noise."""
    if target_col not in df.columns:
        return df, {"injector": "label_noise", "skipped": True}

    rng = get_rng(seed)
    result = df.clone()
    y = result[target_col].to_numpy()

    if not (np.issubdtype(y.dtype, np.integer) and set(np.unique(y)).issubset({0, 1})):
        return df, {"injector": "label_noise", "skipped": True}

    if noise_type == "symmetric":
        mask = rng.uniform(0, 1, size=len(y)) < severity
        new_y = np.where(mask, 1 - y, y)
    elif noise_type == "asymmetric":
        # Only flip class 0 to class 1 (false positive injection)
        mask = (y == 0) & (rng.uniform(0, 1, size=len(y)) < severity)
        new_y = np.where(mask, 1, y)
    else:
        mask = rng.uniform(0, 1, size=len(y)) < severity
        new_y = np.where(mask, 1 - y, y)

    result = result.with_columns(pl.Series(name=target_col, values=new_y))

    return result, {
        "injector": "label_noise",
        "noise_type": noise_type,
        "severity": severity,
        "corrupted_labels": int(np.sum(mask)),
    }


def inject_duplicates(
    df: pl.DataFrame,
    duplicate_ratio: float,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Inject duplicate rows to test data integrity and leakage."""
    rng = get_rng(seed)
    n_duplicates = int(df.height * duplicate_ratio)
    if n_duplicates <= 0:
        return df, {"injector": "duplicates", "n_duplicates": 0}

    sample_indices = rng.choice(df.height, size=n_duplicates, replace=True)
    dupe_df = df[sample_indices]
    result = pl.concat([df, dupe_df])

    return result, {
        "injector": "duplicates",
        "duplicate_ratio": duplicate_ratio,
        "n_duplicates": n_duplicates,
        "total_rows": result.height,
    }


def inject_outliers_and_sentinels(
    df: pl.DataFrame,
    columns: list[str],
    sentinel_values: list[Any] | None = None,
    severity: float = 0.05,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Inject extreme outlier numbers or sentinel markers (-999, 9999)."""
    sentinels = sentinel_values or [-999.0, 9999.0]
    rng = get_rng(seed)
    result = df.clone()
    target_cols = [c for c in columns if c in result.columns]
    affected_cells = 0

    for col in target_cols:
        series = result[col].to_numpy()
        if np.issubdtype(series.dtype, np.number):
            mask = rng.uniform(0, 1, size=len(series)) < severity
            affected_cells += int(np.sum(mask))
            chosen_sentinels = rng.choice(sentinels, size=len(series))
            corrupted = np.where(mask, chosen_sentinels, series)
            result = result.with_columns(pl.Series(name=col, values=corrupted))

    return result, {
        "injector": "outliers_sentinels",
        "severity": severity,
        "columns": target_cols,
        "affected_cells": affected_cells,
    }


def inject_type_corruption(
    df: pl.DataFrame,
    columns: list[str],
    severity: float = 0.05,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Corrupt numeric columns into messy strings (with currency, commas, or 'N/A')."""
    rng = get_rng(seed)
    result = df.clone()
    target_cols = [c for c in columns if c in result.columns]

    for col in target_cols:
        series = result[col].to_numpy()
        mask = rng.uniform(0, 1, size=len(series)) < severity
        corrupted_vals = []
        for val, is_bad in zip(series, mask, strict=False):
            if is_bad:
                variant = rng.choice(["$ {val}", "{val}%", "N/A", "null", "{val} USD"])
                corrupted_vals.append(variant.replace("{val}", str(val)))
            else:
                corrupted_vals.append(str(val))
        result = result.with_columns(pl.Series(name=col, values=corrupted_vals))

    return result, {
        "injector": "type_corruption",
        "severity": severity,
        "columns": target_cols,
    }


def inject_schema_mutation(
    df: pl.DataFrame,
    drop_columns: list[str] | None = None,
    rename_columns: dict[str, str] | None = None,
    add_unexpected_columns: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Inject schema mutations: drop, rename, or add surprise columns."""
    result = df.clone()

    dropped = []
    if drop_columns:
        for c in drop_columns:
            if c in result.columns:
                result = result.drop(c)
                dropped.append(c)

    renamed = {}
    if rename_columns:
        for old_c, new_c in rename_columns.items():
            if old_c in result.columns:
                result = result.rename({old_c: new_c})
                renamed[old_c] = new_c

    added = []
    if add_unexpected_columns:
        for col_name, default_val in add_unexpected_columns.items():
            result = result.with_columns(pl.lit(default_val).alias(col_name))
            added.append(col_name)

    return result, {
        "injector": "schema_mutation",
        "dropped_columns": dropped,
        "renamed_columns": renamed,
        "added_columns": added,
    }
