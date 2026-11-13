"""Atomic, reference-fitted data repair strategies with reversible lineage patches."""

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass
class RepairPatch:
    """Atomic change record capturing cell-level transformation."""

    row_idx: int
    column_name: str
    old_value: Any
    new_value: Any
    strategy: str


@dataclass
class RepairResult:
    repaired_df: pl.DataFrame
    patches: list[RepairPatch]
    cells_modified: int
    fraction_modified: float
    strategy_summary: dict[str, int]


def impute_missing_values(
    df: pl.DataFrame,
    ref_stats: dict[str, Any],
) -> tuple[pl.DataFrame, list[RepairPatch]]:
    """Impute nulls strictly using reference distributions (Fit-on-train discipline)."""
    result = df.clone()
    patches: list[RepairPatch] = []

    # 1. Numeric columns (impute with reference median q_50)
    for col, stats in ref_stats.get("numeric", {}).items():
        if col not in result.columns:
            continue
        null_mask = result[col].is_null().to_numpy()
        if not np.any(null_mask):
            continue

        fill_val = float(stats.get("quantiles", {}).get("q_50", stats.get("mean", 0.0)))
        null_indices = np.where(null_mask)[0]
        for idx in null_indices:
            patches.append(
                RepairPatch(
                    row_idx=int(idx),
                    column_name=col,
                    old_value=None,
                    new_value=fill_val,
                    strategy="impute_numeric_median",
                )
            )

        result = result.with_columns(
            pl.when(pl.col(col).is_null()).then(fill_val).otherwise(pl.col(col)).alias(col)
        )

    # 2. Categorical columns (impute with reference mode)
    for col, stats in ref_stats.get("categorical", {}).items():
        if col not in result.columns:
            continue
        null_mask = result[col].is_null().to_numpy()
        if not np.any(null_mask):
            continue

        freqs = stats.get("frequencies", {})
        fill_cat = max(freqs, key=freqs.get) if freqs else "MISSING"
        null_indices = np.where(null_mask)[0]
        for idx in null_indices:
            patches.append(
                RepairPatch(
                    row_idx=int(idx),
                    column_name=col,
                    old_value=None,
                    new_value=fill_cat,
                    strategy="impute_categorical_mode",
                )
            )

        result = result.with_columns(
            pl.when(pl.col(col).is_null()).then(pl.lit(fill_cat)).otherwise(pl.col(col)).alias(col)
        )

    return result, patches


def winsorize_outliers(
    df: pl.DataFrame,
    ref_stats: dict[str, Any],
) -> tuple[pl.DataFrame, list[RepairPatch]]:
    """Winsorize extreme outliers back to reference distribution quantiles (q_01, q_99)."""
    result = df.clone()
    patches: list[RepairPatch] = []

    for col, stats in ref_stats.get("numeric", {}).items():
        if col not in result.columns:
            continue
        quantiles = stats.get("quantiles", {})
        q01 = float(quantiles.get("q_01", stats.get("min", -1e9)))
        q99 = float(quantiles.get("q_99", stats.get("max", 1e9)))

        series = result[col].to_numpy()
        low_mask = series < q01
        high_mask = series > q99

        for idx in np.where(low_mask)[0]:
            patches.append(
                RepairPatch(
                    row_idx=int(idx),
                    column_name=col,
                    old_value=float(series[idx]),
                    new_value=q01,
                    strategy="winsorize_low",
                )
            )
        for idx in np.where(high_mask)[0]:
            patches.append(
                RepairPatch(
                    row_idx=int(idx),
                    column_name=col,
                    old_value=float(series[idx]),
                    new_value=q99,
                    strategy="winsorize_high",
                )
            )

        if np.any(low_mask) or np.any(high_mask):
            result = result.with_columns(pl.col(col).clip(q01, q99).alias(col))

    return result, patches


def repair_sentinels(
    df: pl.DataFrame,
    ref_stats: dict[str, Any],
    sentinels: list[Any] | None = None,
) -> tuple[pl.DataFrame, list[RepairPatch]]:
    """Detect and replace sentinel values (-999, 9999, 'N/A') with reference medians."""
    sentinel_set = set(sentinels or [-999.0, 9999.0, -999, 9999])
    result = df.clone()
    patches: list[RepairPatch] = []

    for col, stats in ref_stats.get("numeric", {}).items():
        if col not in result.columns:
            continue
        series = result[col].to_numpy()
        mask = np.isin(series, list(sentinel_set))
        if not np.any(mask):
            continue

        median_val = float(stats.get("quantiles", {}).get("q_50", stats.get("mean", 0.0)))
        for idx in np.where(mask)[0]:
            patches.append(
                RepairPatch(
                    row_idx=int(idx),
                    column_name=col,
                    old_value=float(series[idx]),
                    new_value=median_val,
                    strategy="repair_sentinel",
                )
            )

        result = result.with_columns(
            pl.when(pl.col(col).is_in(list(sentinel_set)))
            .then(median_val)
            .otherwise(pl.col(col))
            .alias(col)
        )

    return result, patches


def coerce_types(
    df: pl.DataFrame,
    ref_stats: dict[str, Any],
) -> tuple[pl.DataFrame, list[RepairPatch]]:
    """Clean and cast stringified numeric columns (strip currency, commas, %)."""
    result = df.clone()
    patches: list[RepairPatch] = []

    for col, stats in ref_stats.get("numeric", {}).items():
        if col not in result.columns:
            continue
        if result[col].dtype in (pl.Utf8, pl.String):
            raw_strs = result[col].to_list()
            median_val = float(stats.get("quantiles", {}).get("q_50", stats.get("mean", 0.0)))
            cleaned_vals = []

            for idx, raw in enumerate(raw_strs):
                if raw is None:
                    cleaned_vals.append(median_val)
                    patches.append(
                        RepairPatch(
                            row_idx=idx,
                            column_name=col,
                            old_value=raw,
                            new_value=median_val,
                            strategy="coerce_type_none",
                        )
                    )
                else:
                    # Strip $, %, commas, whitespace
                    clean_str = re.sub(r"[\$,%USD\s]", "", str(raw))
                    try:
                        f_val = float(clean_str)
                    except ValueError:
                        f_val = median_val
                    cleaned_vals.append(f_val)
                    if str(raw) != str(f_val):
                        patches.append(
                            RepairPatch(
                                row_idx=idx,
                                column_name=col,
                                old_value=raw,
                                new_value=f_val,
                                strategy="coerce_type_string_to_float",
                            )
                        )

            result = result.with_columns(pl.Series(name=col, values=cleaned_vals).cast(pl.Float64))

    return result, patches


def map_unseen_categories(
    df: pl.DataFrame,
    ref_stats: dict[str, Any],
) -> tuple[pl.DataFrame, list[RepairPatch]]:
    """Map out-of-vocabulary categories to the most frequent training category or 'OTHER'."""
    result = df.clone()
    patches: list[RepairPatch] = []

    for col, stats in ref_stats.get("categorical", {}).items():
        if col not in result.columns:
            continue
        known_cats = set(stats.get("frequencies", {}).keys())
        if not known_cats:
            continue

        top_cat = max(stats["frequencies"], key=stats["frequencies"].get)
        raw_vals = result[col].to_list()
        new_vals = []

        for idx, val in enumerate(raw_vals):
            if val is not None and str(val) not in known_cats:
                new_vals.append(top_cat)
                patches.append(
                    RepairPatch(
                        row_idx=idx,
                        column_name=col,
                        old_value=val,
                        new_value=top_cat,
                        strategy="map_unseen_category",
                    )
                )
            else:
                new_vals.append(val)

        if len(patches) > 0:
            result = result.with_columns(pl.Series(name=col, values=new_vals))

    return result, patches


def realign_schema(
    df: pl.DataFrame,
    ref_stats: dict[str, Any],
    expected_features: list[str],
) -> tuple[pl.DataFrame, list[RepairPatch]]:
    """Fill missing expected columns with reference median/mode and prune unexpected extra columns."""
    result = df.clone()
    patches: list[RepairPatch] = []

    # 1. Fill missing columns
    for col in expected_features:
        if col not in result.columns:
            if col in ref_stats.get("numeric", {}):
                fill_val = float(
                    ref_stats["numeric"][col]
                    .get("quantiles", {})
                    .get("q_50", ref_stats["numeric"][col].get("mean", 0.0))
                )
            elif col in ref_stats.get("categorical", {}):
                freqs = ref_stats["categorical"][col].get("frequencies", {})
                fill_val = max(freqs, key=freqs.get) if freqs else "UNKNOWN"
            else:
                fill_val = 0.0

            result = result.with_columns(pl.lit(fill_val).alias(col))
            patches.append(
                RepairPatch(
                    row_idx=-1,  # Entire column
                    column_name=col,
                    old_value="[COLUMN_MISSING]",
                    new_value=fill_val,
                    strategy="realign_schema_add_missing_column",
                )
            )

    return result, patches
