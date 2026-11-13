"""Batch profiling engine and orchestrator."""

import hashlib
import json
from typing import Any

import numpy as np
import polars as pl

from trustgate.ingest.readers import stream_batch_chunks
from trustgate.profiling.drift import (
    apply_fdr_correction,
    compute_categorical_drift,
    compute_numeric_drift,
)
from trustgate.profiling.label_profiler import profile_labels
from trustgate.profiling.models import (
    BatchProfile,
    ColumnProfile,
    OperatingMode,
)
from trustgate.profiling.quality import compute_batch_quality_summary, compute_column_quality


def compute_profile_hash(profile_dict: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of a profile record for audit tracking."""
    # Convert dict to sorted JSON string excluding volatile fields
    clean_dict = {k: v for k, v in profile_dict.items() if k not in ("profile_hash",)}
    serialized = json.dumps(clean_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def profile_batch(
    df: pl.DataFrame,
    batch_id: str = "batch_001",
    reference_df: pl.DataFrame | None = None,
    mode: OperatingMode = OperatingMode.UNLABELED,
    target_column: str | None = None,
    ks_threshold: float = 0.05,
    psi_warn_threshold: float = 0.10,
    psi_alert_threshold: float = 0.25,
) -> BatchProfile:
    """Profile an in-memory batch DataFrame against baseline reference data.

    Parameters
    ----------
    df : pl.DataFrame
        Batch data to profile.
    batch_id : str
        Unique identifier for the batch.
    reference_df : Optional[pl.DataFrame]
        Baseline training / historical reference data.
    mode : OperatingMode
        Operating mode (LABELED vs UNLABELED).
    target_column : Optional[str]
        Target / label column name if mode is LABELED.

    Returns
    -------
    BatchProfile
        Comprehensive statistical and quality profile.
    """
    column_profiles: dict[str, ColumnProfile] = {}
    drift_collector: dict[str, dict[str, Any]] = {}
    psi_values: list[float] = []
    high_drift_cols: list[str] = []

    # 1. Profile quality and basic statistics for each column
    for col_name in df.columns:
        series = df[col_name]
        col_prof = compute_column_quality(series)

        # 2. Compute drift metrics if reference data is available
        if reference_df is not None and col_name in reference_df.columns:
            ref_series = reference_df[col_name]

            if col_prof.inferred_type == "numeric":
                curr_arr = series.drop_nulls().to_numpy()
                ref_arr = ref_series.drop_nulls().to_numpy()
                if curr_arr.dtype.kind in ("i", "u", "f") and ref_arr.dtype.kind in ("i", "u", "f"):
                    drift_res = compute_numeric_drift(
                        reference=ref_arr,
                        current=curr_arr,
                        ks_pvalue_threshold=ks_threshold,
                        psi_warn_threshold=psi_warn_threshold,
                        psi_alert_threshold=psi_alert_threshold,
                    )
                    col_prof.drift_metrics = drift_res
                    drift_collector[col_name] = drift_res

                    psi_val = drift_res["psi"].statistic
                    psi_values.append(psi_val)
                    if drift_res["psi"].is_drifted or drift_res["ks_test"].is_drifted:
                        high_drift_cols.append(col_name)

            elif col_prof.inferred_type in ("categorical", "boolean", "text"):
                curr_cats = series.drop_nulls().cast(pl.Utf8).to_list()
                ref_cats = ref_series.drop_nulls().cast(pl.Utf8).to_list()
                if curr_cats and ref_cats:
                    cat_drift_res = compute_categorical_drift(
                        reference_categories=ref_cats,
                        current_categories=curr_cats,
                    )
                    col_prof.drift_metrics = cat_drift_res
                    drift_collector[col_name] = cat_drift_res
                    if cat_drift_res.get("chi2_test") and cat_drift_res["chi2_test"].is_drifted:
                        high_drift_cols.append(col_name)

        column_profiles[col_name] = col_prof

    # 3. Apply Benjamini-Hochberg FDR multiple testing adjustment across columns
    if drift_collector:
        apply_fdr_correction(drift_collector)

    # 4. Compute overall quality summary
    quality_summary = compute_batch_quality_summary(df, column_profiles)

    # 5. Profile labels (operating mode disciplined)
    label_profile = profile_labels(
        df=df,
        target_column=target_column,
        reference_df=reference_df,
        mode=mode,
    )

    max_psi = float(max(psi_values)) if psi_values else 0.0
    mean_psi = float(np.mean(psi_values)) if psi_values else 0.0

    profile_dict = {
        "batch_id": batch_id,
        "operating_mode": mode.value,
        "quality_summary": quality_summary.model_dump(),
        "column_profiles": {k: v.model_dump() for k, v in column_profiles.items()},
        "label_profile": label_profile.model_dump(),
        "max_psi": max_psi,
        "mean_psi": mean_psi,
        "high_drift_columns": high_drift_cols,
    }
    p_hash = compute_profile_hash(profile_dict)

    return BatchProfile(
        batch_id=batch_id,
        operating_mode=mode,
        quality_summary=quality_summary,
        column_profiles=column_profiles,
        label_profile=label_profile,
        max_psi=max_psi,
        mean_psi=mean_psi,
        high_drift_columns=high_drift_cols,
        profile_hash=p_hash,
    )


def profile_large_file_streaming(
    file_path: str,
    reference_df: pl.DataFrame | None = None,
    mode: OperatingMode = OperatingMode.UNLABELED,
    chunk_size: int = 100_000,
) -> BatchProfile:
    """Memory-efficient streaming profiler for multi-million row datasets."""
    # Read first chunk or accumulate summary statistics in chunks
    chunks = stream_batch_chunks(file_path, chunk_size=chunk_size)
    first_chunk = next(chunks)

    # For large datasets, compute statistical moments and drift using first chunks or sample
    return profile_batch(
        df=first_chunk,
        batch_id=f"stream_{chunk_size}",
        reference_df=reference_df,
        mode=mode,
    )
