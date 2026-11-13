"""Unit tests for profiling: statistical drift, quality summary, and label profiling."""

import numpy as np
import polars as pl

from trustgate.profiling.drift import (
    apply_fdr_correction,
    calculate_psi,
    compute_categorical_drift,
    compute_numeric_drift,
)
from trustgate.profiling.label_profiler import profile_labels
from trustgate.profiling.models import MetricStatus, OperatingMode
from trustgate.profiling.profiler import profile_batch


def test_psi_identical_and_drifted() -> None:
    """Verify PSI is near 0 for identical distributions and high for drifted distributions."""
    np.random.seed(42)
    ref = np.random.normal(loc=0.0, scale=1.0, size=1000)
    curr_same = np.random.normal(loc=0.0, scale=1.0, size=1000)
    curr_drifted = np.random.normal(loc=2.5, scale=1.0, size=1000)

    psi_same, _ = calculate_psi(ref, curr_same)
    psi_drifted, _ = calculate_psi(ref, curr_drifted)

    # Identical should have PSI < 0.10
    assert psi_same < 0.10
    # Heavily shifted distribution should have high PSI (>= 0.25)
    assert psi_drifted >= 0.25


def test_drift_zero_variance_and_small_batch() -> None:
    """Verify zero-variance guards and small sample size INSUFFICIENT_DATA status."""
    # Zero variance (constant columns)
    ref_const = np.zeros(100)
    curr_const = np.zeros(100)
    psi_const, _ = calculate_psi(ref_const, curr_const)
    assert psi_const == 0.0

    # Small batch (< 30 rows)
    ref_small = np.random.rand(20)
    curr_small = np.random.rand(20)
    res = compute_numeric_drift(ref_small, curr_small, min_sample_size=30)
    assert res["ks_test"].status == MetricStatus.INSUFFICIENT_DATA
    assert res["psi"].status == MetricStatus.INSUFFICIENT_DATA


def test_categorical_drift_and_new_categories() -> None:
    """Verify Chi2 test and Cramér's V on shifted categorical distributions."""
    ref_cats = ["A"] * 500 + ["B"] * 500
    # Batch has introduced new category "C" and altered balance
    curr_cats = ["A"] * 100 + ["B"] * 200 + ["C"] * 700

    res = compute_categorical_drift(ref_cats, curr_cats)
    assert "chi2_test" in res
    assert res["chi2_test"].is_drifted is True
    assert res["chi2_test"].effect_size > 0.15  # Cramér's V


def test_fdr_correction_multi_column() -> None:
    """Verify Benjamini-Hochberg FDR adjusts p-values across multiple columns."""
    from trustgate.profiling.models import DriftMetricResult

    # Simulate 10 columns where only 1 has a very small p-value
    drift_dict = {
        f"col_{i}": {
            "ks_test": DriftMetricResult(
                test_name="ks",
                statistic=0.1,
                p_value=0.04,  # Marginally significant before FDR
                effect_size=0.1,
            )
        }
        for i in range(10)
    }

    apply_fdr_correction(drift_dict, alpha=0.05)

    # Corrected p-values must be adjusted upwards
    for _col, metrics in drift_dict.items():
        assert metrics["ks_test"].p_value_adjusted is not None
        assert metrics["ks_test"].p_value_adjusted >= 0.04


def test_operating_mode_discipline() -> None:
    """Verify that Unlabeled mode marks label checks as NOT_APPLICABLE and does not fail."""
    df = pl.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [10.0, 20.0, 30.0],
        }
    )

    profile_unlabeled = profile_batch(
        df=df,
        batch_id="inf_001",
        mode=OperatingMode.UNLABELED,
    )

    assert profile_unlabeled.label_profile.is_labeled is False
    assert profile_unlabeled.label_profile.status == MetricStatus.NOT_APPLICABLE


def test_labeled_batch_single_class_edge_case() -> None:
    """Verify that a batch with only 1 class is flagged as WARNING with explicit notes."""
    df_single_class = pl.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 0, 0, 0],  # Single class
        }
    )

    label_prof = profile_labels(
        df=df_single_class,
        target_column="target",
        mode=OperatingMode.LABELED,
    )

    assert label_prof.is_labeled is True
    assert label_prof.num_classes == 1
    assert "Single-class batch detected" in str(label_prof.notes)
    assert label_prof.status == MetricStatus.WARNING
