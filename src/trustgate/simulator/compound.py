"""Compound corruptions orchestrating 2 to 4 simultaneous interacting issues."""

from typing import Any

import polars as pl

from trustgate.simulator.datasets import DatasetSpec
from trustgate.simulator.injectors import (
    inject_label_noise,
    inject_mar,
    inject_mcar,
    inject_mean_shift,
    inject_outliers_and_sentinels,
    inject_unit_error,
    inject_unseen_categories,
    inject_variance_scale,
)
from trustgate.utils.seeding import get_rng


def inject_compound_corruption(
    df: pl.DataFrame,
    spec: DatasetSpec,
    severity: float = 0.1,
    num_issues: int = 3,
    seed: int = 42,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Inject 2 to 4 simultaneous interacting data quality issues."""
    rng = get_rng(seed)
    current_df = df.clone()
    applied_logs: list[dict[str, Any]] = []

    available_actions = [
        "missingness",
        "mean_shift",
        "variance_scale",
        "outliers",
        "unseen_categories",
        "unit_error",
    ]
    if spec.task_type == "classification":
        available_actions.append("label_noise")

    k = min(len(available_actions), max(2, min(4, num_issues)))
    chosen_actions = list(rng.choice(available_actions, size=k, replace=False))

    for idx, action in enumerate(chosen_actions):
        sub_seed = seed + (idx + 1) * 101

        if action == "missingness":
            if len(spec.numeric_features) >= 2 and rng.uniform(0, 1) > 0.5:
                # Apply MAR
                target = str(rng.choice(spec.numeric_features))
                cond = str(rng.choice([c for c in spec.numeric_features if c != target]))
                current_df, log = inject_mar(
                    current_df,
                    target_col=target,
                    condition_col=cond,
                    severity=severity,
                    seed=sub_seed,
                )
            else:
                # Apply MCAR
                cols = list(
                    rng.choice(
                        spec.all_features, size=min(2, len(spec.all_features)), replace=False
                    )
                )
                current_df, log = inject_mcar(
                    current_df,
                    columns=cols,
                    severity=severity,
                    seed=sub_seed,
                )
            applied_logs.append(log)

        elif action == "mean_shift":
            cols = list(
                rng.choice(
                    spec.numeric_features, size=min(2, len(spec.numeric_features)), replace=False
                )
            )
            shift_stds = float(rng.choice([0.5, 1.0, 1.5, -1.0]))
            current_df, log = inject_mean_shift(
                current_df,
                columns=cols,
                shift_stds=shift_stds,
                seed=sub_seed,
            )
            applied_logs.append(log)

        elif action == "variance_scale":
            cols = list(rng.choice(spec.numeric_features, size=1, replace=False))
            scale_factor = float(rng.choice([0.5, 2.0, 3.0]))
            current_df, log = inject_variance_scale(
                current_df,
                columns=cols,
                scale_factor=scale_factor,
                seed=sub_seed,
            )
            applied_logs.append(log)

        elif action == "outliers":
            cols = list(rng.choice(spec.numeric_features, size=1, replace=False))
            current_df, log = inject_outliers_and_sentinels(
                current_df,
                columns=cols,
                severity=severity * 0.5,
                seed=sub_seed,
            )
            applied_logs.append(log)

        elif action == "unseen_categories" and spec.categorical_features:
            cat_col = str(rng.choice(spec.categorical_features))
            current_df, log = inject_unseen_categories(
                current_df,
                column=cat_col,
                severity=severity,
                seed=sub_seed,
            )
            applied_logs.append(log)

        elif action == "unit_error":
            col = str(rng.choice(spec.numeric_features))
            mult = float(rng.choice([10.0, 100.0, 0.01]))
            current_df, log = inject_unit_error(
                current_df,
                column=col,
                multiplier=mult,
                seed=sub_seed,
            )
            applied_logs.append(log)

        elif action == "label_noise":
            current_df, log = inject_label_noise(
                current_df,
                target_col=spec.target_column,
                severity=severity,
                seed=sub_seed,
            )
            applied_logs.append(log)

    return current_df, applied_logs
