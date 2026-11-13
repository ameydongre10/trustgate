"""Statistical drift tests: Kolmogorov-Smirnov, PSI with epsilon-smoothing, Chi2/Cramér's V, and FDR."""

from typing import Any

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from trustgate.profiling.models import DriftMetricResult, MetricStatus


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> tuple[float, dict[str, Any]]:
    """Calculate Population Stability Index (PSI) with epsilon-smoothing and tie handling.

    Parameters
    ----------
    reference : np.ndarray
        Baseline reference values (must be 1D, non-empty, finite).
    current : np.ndarray
        Current batch values (must be 1D, non-empty, finite).
    num_bins : int
        Number of quantile bins (default 10).
    epsilon : float
        Laplace smoothing factor to eliminate division by zero and log(0).

    Returns
    -------
    tuple[float, dict[str, Any]]
        (psi_value, bin_details)
    """
    ref_clean = reference[np.isfinite(reference)]
    curr_clean = current[np.isfinite(current)]

    if len(ref_clean) < 10 or len(curr_clean) < 10:
        return 0.0, {"status": "INSUFFICIENT_DATA"}

    # Zero-variance guard: if both reference and current have zero variance
    ref_std = float(np.std(ref_clean))
    curr_std = float(np.std(curr_clean))
    if ref_std == 0.0 and curr_std == 0.0:
        # If identical constants, PSI is 0; if different constants, significant drift
        return (0.0 if float(np.mean(ref_clean)) == float(np.mean(curr_clean)) else 1.0), {
            "status": "ZERO_VARIANCE"
        }

    # Compute quantile bin edges on reference data
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    # Deduplicate bin edges in case of heavy ties (e.g. lots of zeros)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0, {"status": "INSUFFICIENT_UNIQUE_BINS"}

    # Extend boundaries to infinity to capture out-of-range values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Compute counts in each bin
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    # Convert to proportions with epsilon smoothing
    ref_pct = (ref_counts + epsilon) / (np.sum(ref_counts) + epsilon * len(ref_counts))
    curr_pct = (curr_counts + epsilon) / (np.sum(curr_counts) + epsilon * len(curr_counts))

    # PSI calculation: sum( (Actual - Expected) * ln(Actual / Expected) )
    psi_vector = (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
    # Guard against NaN/Inf from floating point inaccuracies
    psi_vector = np.nan_to_num(psi_vector, nan=0.0, posinf=0.0, neginf=0.0)
    psi_value = float(np.sum(psi_vector))

    # Clip to >= 0 mathematically
    psi_value = max(0.0, psi_value)

    return psi_value, {
        "num_bins_used": len(bin_edges) - 1,
        "ref_samples": len(ref_clean),
        "curr_samples": len(curr_clean),
    }


def compute_numeric_drift(
    reference: np.ndarray,
    current: np.ndarray,
    ks_pvalue_threshold: float = 0.05,
    psi_warn_threshold: float = 0.10,
    psi_alert_threshold: float = 0.25,
    min_sample_size: int = 30,
) -> dict[str, DriftMetricResult]:
    """Compute two-sample Kolmogorov-Smirnov test and PSI for numeric distributions."""
    ref_clean = reference[np.isfinite(reference)]
    curr_clean = current[np.isfinite(current)]

    results: dict[str, DriftMetricResult] = {}

    # Check minimum sample size constraint
    if len(ref_clean) < min_sample_size or len(curr_clean) < min_sample_size:
        results["ks_test"] = DriftMetricResult(
            test_name="kolmogorov_smirnov",
            statistic=0.0,
            p_value=None,
            effect_size=0.0,
            is_drifted=False,
            status=MetricStatus.INSUFFICIENT_DATA,
            details={
                "sample_size": min(len(ref_clean), len(curr_clean)),
                "min_required": min_sample_size,
            },
        )
        results["psi"] = DriftMetricResult(
            test_name="population_stability_index",
            statistic=0.0,
            effect_size=0.0,
            is_drifted=False,
            status=MetricStatus.INSUFFICIENT_DATA,
        )
        return results

    # 1. Kolmogorov-Smirnov two-sample test
    ks_res = stats.ks_2samp(ref_clean, curr_clean)
    ks_stat = float(ks_res.statistic)
    ks_pvalue = float(ks_res.pvalue)

    results["ks_test"] = DriftMetricResult(
        test_name="kolmogorov_smirnov",
        statistic=ks_stat,
        p_value=ks_pvalue,
        effect_size=ks_stat,
        is_drifted=(ks_pvalue < ks_pvalue_threshold and ks_stat > 0.05),
        status=MetricStatus.DRIFT_DETECTED
        if (ks_pvalue < ks_pvalue_threshold and ks_stat > 0.05)
        else MetricStatus.OK,
    )

    # 2. Population Stability Index (PSI)
    psi_val, psi_details = calculate_psi(ref_clean, curr_clean)
    is_drifted_psi = psi_val >= psi_alert_threshold
    status_psi = (
        MetricStatus.DRIFT_DETECTED
        if is_drifted_psi
        else (MetricStatus.WARNING if psi_val >= psi_warn_threshold else MetricStatus.OK)
    )

    results["psi"] = DriftMetricResult(
        test_name="population_stability_index",
        statistic=psi_val,
        effect_size=psi_val,
        is_drifted=is_drifted_psi,
        status=status_psi,
        details=psi_details,
    )

    return results


def compute_categorical_drift(
    reference_categories: list[str],
    current_categories: list[str],
    chi2_pvalue_threshold: float = 0.05,
    cramers_v_threshold: float = 0.15,
    min_sample_size: int = 30,
) -> dict[str, DriftMetricResult]:
    """Compute Chi-square goodness-of-fit test and Cramér's V effect size for categorical features."""
    results: dict[str, DriftMetricResult] = {}

    if len(reference_categories) < min_sample_size or len(current_categories) < min_sample_size:
        results["chi2_test"] = DriftMetricResult(
            test_name="chi_square_test",
            statistic=0.0,
            p_value=None,
            effect_size=0.0,
            is_drifted=False,
            status=MetricStatus.INSUFFICIENT_DATA,
        )
        return results

    # Compute category distributions
    ref_counts: dict[str, int] = {}
    for c in reference_categories:
        ref_counts[c] = ref_counts.get(c, 0) + 1

    curr_counts: dict[str, int] = {}
    for c in current_categories:
        curr_counts[c] = curr_counts.get(c, 0) + 1

    all_keys = set(ref_counts.keys()).union(set(curr_counts.keys()))
    if len(all_keys) <= 1:
        results["chi2_test"] = DriftMetricResult(
            test_name="chi_square_test",
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            is_drifted=False,
            status=MetricStatus.OK,
        )
        return results

    # Align frequency vectors with smoothing for rare/unseen categories
    ref_vec = np.array([ref_counts.get(k, 0) for k in all_keys], dtype=float)
    curr_vec = np.array([curr_counts.get(k, 0) for k in all_keys], dtype=float)

    ref_prob = (ref_vec + 1e-4) / (np.sum(ref_vec) + 1e-4 * len(all_keys))
    # Expected counts based on current batch size
    expected = ref_prob * np.sum(curr_vec)
    observed = curr_vec

    try:
        chi2_stat, p_val = stats.chisquare(f_obs=observed, f_exp=expected)
        chi2_stat = float(chi2_stat)
        p_val = float(p_val)

        # Cramér's V effect size
        n: float = float(np.sum(observed))
        k = len(all_keys)
        cramers_v = np.sqrt(chi2_stat / (n * max(1, k - 1))) if n > 0 and k > 1 else 0.0
        cramers_v = float(min(1.0, max(0.0, cramers_v)))

        is_drifted = p_val < chi2_pvalue_threshold and cramers_v > cramers_v_threshold

        results["chi2_test"] = DriftMetricResult(
            test_name="chi_square_test",
            statistic=chi2_stat,
            p_value=p_val,
            effect_size=cramers_v,
            is_drifted=is_drifted,
            status=MetricStatus.DRIFT_DETECTED if is_drifted else MetricStatus.OK,
            details={"cramers_v": cramers_v, "num_categories": k},
        )
    except Exception as e:
        results["chi2_test"] = DriftMetricResult(
            test_name="chi_square_test",
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            is_drifted=False,
            status=MetricStatus.OK,
            details={"error": str(e)},
        )

    return results


def apply_fdr_correction(
    drift_results: dict[str, dict[str, DriftMetricResult]], alpha: float = 0.05
) -> None:
    """Apply Benjamini-Hochberg False Discovery Rate (FDR) adjustment across all p-values."""
    p_values: list[float] = []
    metric_refs: list[DriftMetricResult] = []

    for _col, metrics in drift_results.items():
        for _metric_name, res in metrics.items():
            if res.p_value is not None and np.isfinite(res.p_value):
                p_values.append(res.p_value)
                metric_refs.append(res)

    if not p_values:
        return

    # Apply Benjamini-Hochberg FDR
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

    for i, res in enumerate(metric_refs):
        res.p_value_adjusted = float(pvals_corrected[i])
        # Update is_drifted based on FDR-adjusted p-value paired with effect size
        if res.effect_size > 0.05:
            res.is_drifted = bool(reject[i])
            if res.is_drifted:
                res.status = MetricStatus.DRIFT_DETECTED
