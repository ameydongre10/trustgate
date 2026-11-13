"""Label profiling for training/retraining batches vs. inference batches."""

import numpy as np
import polars as pl
from scipy import stats

from trustgate.profiling.models import LabelProfile, MetricStatus, OperatingMode


def profile_labels(
    df: pl.DataFrame,
    target_column: str | None = None,
    reference_df: pl.DataFrame | None = None,
    mode: OperatingMode = OperatingMode.UNLABELED,
) -> LabelProfile:
    """Profile label distributions and detect label noise / shift.

    Parameters
    ----------
    df : pl.DataFrame
        Batch DataFrame.
    target_column : Optional[str]
        Name of the target / label column.
    reference_df : Optional[pl.DataFrame]
        Reference training DataFrame.
    mode : OperatingMode
        Operating mode (LABELED or UNLABELED).

    Returns
    -------
    LabelProfile
        Evaluated label profile.
    """
    if mode == OperatingMode.UNLABELED or not target_column or target_column not in df.columns:
        return LabelProfile(
            is_labeled=False,
            target_column=target_column,
            status=MetricStatus.NOT_APPLICABLE,
            notes="Unlabeled inference batch: label-dependent checks marked NOT_APPLICABLE.",
        )

    series = df[target_column].drop_nulls()
    if len(series) == 0:
        return LabelProfile(
            is_labeled=True,
            target_column=target_column,
            status=MetricStatus.WARNING,
            notes="Target column is completely null in batch.",
        )

    # Compute class balance
    value_counts = series.value_counts()
    total_samples = len(series)
    class_balance: dict[str, float] = {}

    for row in value_counts.iter_rows():
        label_val = str(row[0])
        count = int(row[1])
        class_balance[label_val] = count / total_samples

    num_classes = len(class_balance)
    minority_rate = min(class_balance.values()) if class_balance else 0.0

    notes = []
    # Check for single-class batch edge case (AUC undefined)
    if num_classes == 1:
        notes.append("Single-class batch detected: downstream AUC/PR-AUC metric is undefined.")

    # Check for extreme class imbalance (< 0.1% positives)
    if minority_rate < 0.001 and num_classes > 1:
        notes.append(
            f"Extreme class imbalance detected: minority class rate is {minority_rate:.4%}."
        )

    label_shift = False
    shift_p_val = None

    # Compare with reference target distribution if available
    if reference_df is not None and target_column in reference_df.columns:
        ref_series = reference_df[target_column].drop_nulls()
        ref_counts_map = {str(k): v for k, v in ref_series.value_counts().iter_rows()}

        all_labels = set(class_balance.keys()).union(set(ref_counts_map.keys()))
        if len(all_labels) > 1:
            ref_total = len(ref_series)
            ref_probs = np.array([ref_counts_map.get(lbl, 0) / ref_total for lbl in all_labels])
            # Laplace smoothing
            ref_probs = (ref_probs + 1e-4) / (np.sum(ref_probs) + 1e-4 * len(all_labels))

            curr_counts_arr = np.array(
                [
                    value_counts.filter(pl.col(target_column) == lbl)["count"].item()
                    if lbl in class_balance
                    else 0
                    for lbl in all_labels
                ]
            )
            expected = ref_probs * np.sum(curr_counts_arr)

            try:
                _, p_val = stats.chisquare(f_obs=curr_counts_arr, f_exp=expected)
                shift_p_val = float(p_val)
                if shift_p_val < 0.01:
                    label_shift = True
                    notes.append(
                        f"Significant label distribution shift detected (p={shift_p_val:.4e})."
                    )
            except Exception:
                pass

    return LabelProfile(
        is_labeled=True,
        target_column=target_column,
        class_balance=class_balance,
        num_classes=num_classes,
        minority_class_rate=minority_rate,
        label_shift_detected=label_shift,
        label_shift_p_value=shift_p_val,
        estimated_noise_rate=0.0,
        status=MetricStatus.DRIFT_DETECTED
        if label_shift
        else (MetricStatus.WARNING if num_classes == 1 else MetricStatus.OK),
        notes="; ".join(notes) if notes else "Label distribution verified.",
    )
