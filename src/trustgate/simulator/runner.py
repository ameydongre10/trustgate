"""Experiment runner generating training data for damage prediction across datasets, models, and corruptions."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from trustgate.profiling.drift import calculate_psi
from trustgate.simulator.compound import inject_compound_corruption
from trustgate.simulator.datasets import DatasetSpec, get_dataset
from trustgate.simulator.injectors import (
    inject_concept_drift,
    inject_duplicates,
    inject_label_noise,
    inject_mar,
    inject_mcar,
    inject_mean_shift,
    inject_mnar,
    inject_outliers_and_sentinels,
    inject_type_corruption,
    inject_unit_error,
    inject_unseen_categories,
    inject_variance_scale,
)
from trustgate.simulator.models import TargetModel, train_target_model
from trustgate.utils.seeding import get_rng, set_global_seed


def extract_fast_issue_features(
    corrupted_df: pl.DataFrame,
    clean_ref_df: pl.DataFrame,
    target_model: TargetModel,
    spec: DatasetSpec,
) -> dict[str, float]:
    """Extract model-aware issue profile features for a corrupted batch."""
    features: dict[str, float] = {}
    n_rows = corrupted_df.height
    features["batch_size"] = float(n_rows)
    features["n_columns"] = float(len(corrupted_df.columns))

    # 1. Null rates
    null_rates: dict[str, float] = {}
    for col in spec.all_features:
        if col in corrupted_df.columns:
            null_rates[col] = float(corrupted_df[col].null_count() / max(1, n_rows))
        else:
            null_rates[col] = 1.0  # Missing column completely

    features["max_null_rate"] = float(max(null_rates.values())) if null_rates else 0.0
    features["mean_null_rate"] = float(np.mean(list(null_rates.values()))) if null_rates else 0.0
    features["importance_weighted_null_rate"] = target_model.get_importance_weighted_vector(
        null_rates
    )

    # 2. Covariate drift (PSI and KS on numerics)
    psi_dict: dict[str, float] = {}
    ks_dict: dict[str, float] = {}
    outlier_rates: list[float] = []

    for col in spec.numeric_features:
        if col in corrupted_df.columns and col in clean_ref_df.columns:
            corr_s = corrupted_df[col].drop_nulls().to_numpy()
            ref_s = clean_ref_df[col].drop_nulls().to_numpy()

            if len(corr_s) > 10 and len(ref_s) > 10:
                psi_val, _ = calculate_psi(ref_s, corr_s)
                psi_dict[col] = psi_val

                ks_res = stats.ks_2samp(ref_s, corr_s)
                ks_dict[col] = float(ks_res.statistic)

                # Simple z-score outlier rate relative to reference
                ref_mean = np.mean(ref_s)
                ref_std = np.std(ref_s) if np.std(ref_s) > 0 else 1.0
                outliers = np.abs((corr_s - ref_mean) / ref_std) > 3.0
                outlier_rates.append(float(np.mean(outliers)))
            else:
                psi_dict[col] = 0.0
                ks_dict[col] = 0.0
                outlier_rates.append(0.0)
        else:
            psi_dict[col] = 2.0  # Max penalty for missing column
            ks_dict[col] = 1.0
            outlier_rates.append(0.0)

    features["max_psi"] = float(max(psi_dict.values())) if psi_dict else 0.0
    features["mean_psi"] = float(np.mean(list(psi_dict.values()))) if psi_dict else 0.0
    features["importance_weighted_psi"] = target_model.get_importance_weighted_vector(psi_dict)
    features["max_ks_stat"] = float(max(ks_dict.values())) if ks_dict else 0.0
    features["outlier_rate"] = float(np.mean(outlier_rates)) if outlier_rates else 0.0

    # 3. Duplicate rate
    try:
        unique_count = corrupted_df.unique().height
        features["duplicate_rate"] = float((n_rows - unique_count) / max(1, n_rows))
    except Exception:
        features["duplicate_rate"] = 0.0

    # 4. Class balance / Label stats if labeled batch
    if spec.target_column in corrupted_df.columns and spec.task_type == "classification":
        y_corr = corrupted_df[spec.target_column].drop_nulls().to_numpy()
        if len(y_corr) > 0 and np.issubdtype(y_corr.dtype, np.integer):
            pos_ratio = float(np.mean(y_corr == 1))
            features["class_balance_ratio"] = pos_ratio
        else:
            features["class_balance_ratio"] = 0.5
    else:
        features["class_balance_ratio"] = 0.5

    # 5. Schema difference
    features["schema_columns_missing"] = float(
        len(set(spec.all_features) - set(corrupted_df.columns))
    )
    features["schema_columns_extra"] = float(
        len(set(corrupted_df.columns) - set(spec.all_features) - {spec.target_column})
    )

    return features


class ExperimentRunner:
    """Orchestrates generation of corruption simulation data for damage prediction."""

    def __init__(
        self,
        output_dir: str = "data/experiments",
        seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        set_global_seed(seed)

    def run_single_simulation(
        self,
        spec: DatasetSpec,
        target_model: TargetModel,
        clean_ref_df: pl.DataFrame,
        test_df: pl.DataFrame,
        corruption_type: str,
        severity: float,
        seed: int,
    ) -> dict[str, Any]:
        """Apply a single corruption recipe, compute issue profile, evaluate damage."""
        t0 = time.perf_counter()
        rng = get_rng(seed)

        # Baseline clean evaluation
        y_true = test_df[spec.target_column].to_numpy()
        clean_metric = target_model.evaluate(test_df, y_true)

        corrupted_df = test_df.clone()
        metadata: dict[str, Any] = {}

        # Select columns to corrupt
        if corruption_type == "clean":
            pass
        elif corruption_type == "mcar":
            cols = list(
                rng.choice(spec.all_features, size=min(2, len(spec.all_features)), replace=False)
            )
            corrupted_df, metadata = inject_mcar(
                corrupted_df, columns=cols, severity=severity, seed=seed
            )
        elif corruption_type == "mar":
            if len(spec.numeric_features) >= 2:
                target = str(rng.choice(spec.numeric_features))
                cond = str(rng.choice([c for c in spec.numeric_features if c != target]))
                corrupted_df, metadata = inject_mar(
                    corrupted_df,
                    target_col=target,
                    condition_col=cond,
                    severity=severity,
                    seed=seed,
                )
        elif corruption_type == "mnar":
            target = str(rng.choice(spec.numeric_features))
            corrupted_df, metadata = inject_mnar(
                corrupted_df, target_col=target, severity=severity, seed=seed
            )
        elif corruption_type == "mean_shift":
            cols = list(
                rng.choice(
                    spec.numeric_features, size=min(2, len(spec.numeric_features)), replace=False
                )
            )
            corrupted_df, metadata = inject_mean_shift(
                corrupted_df, columns=cols, shift_stds=severity * 3.0, seed=seed
            )
        elif corruption_type == "variance_scale":
            cols = list(rng.choice(spec.numeric_features, size=1, replace=False))
            corrupted_df, metadata = inject_variance_scale(
                corrupted_df, columns=cols, scale_factor=severity * 4.0 + 0.1, seed=seed
            )
        elif corruption_type == "unit_error":
            col = str(rng.choice(spec.numeric_features))
            mult = float(rng.choice([10.0, 100.0, 0.01]))
            corrupted_df, metadata = inject_unit_error(
                corrupted_df, column=col, multiplier=mult, seed=seed
            )
        elif corruption_type == "unseen_categories" and spec.categorical_features:
            cat_col = str(rng.choice(spec.categorical_features))
            corrupted_df, metadata = inject_unseen_categories(
                corrupted_df, column=cat_col, severity=severity, seed=seed
            )
        elif corruption_type == "concept_drift":
            cond_col = str(rng.choice(spec.numeric_features))
            corrupted_df, metadata = inject_concept_drift(
                corrupted_df,
                target_col=spec.target_column,
                condition_col=cond_col,
                severity=severity,
                seed=seed,
            )
        elif corruption_type == "label_noise":
            corrupted_df, metadata = inject_label_noise(
                corrupted_df, target_col=spec.target_column, severity=severity, seed=seed
            )
        elif corruption_type == "duplicates":
            corrupted_df, metadata = inject_duplicates(
                corrupted_df, duplicate_ratio=severity, seed=seed
            )
        elif corruption_type == "outliers":
            cols = list(rng.choice(spec.numeric_features, size=1, replace=False))
            corrupted_df, metadata = inject_outliers_and_sentinels(
                corrupted_df, columns=cols, severity=severity * 0.3, seed=seed
            )
        elif corruption_type == "type_corruption":
            cols = list(rng.choice(spec.numeric_features, size=1, replace=False))
            corrupted_df, metadata = inject_type_corruption(
                corrupted_df, columns=cols, severity=severity * 0.2, seed=seed
            )
        elif corruption_type == "compound":
            corrupted_df, _ = inject_compound_corruption(
                corrupted_df, spec=spec, severity=severity, seed=seed
            )

        # Extract issue profile features
        issue_features = extract_fast_issue_features(
            corrupted_df=corrupted_df,
            clean_ref_df=clean_ref_df,
            target_model=target_model,
            spec=spec,
        )

        # Downstream evaluation on corrupted data
        # Note: If target column was corrupted, evaluation uses corrupted target;
        # otherwise evaluation uses true label
        eval_y = (
            corrupted_df[spec.target_column].to_numpy()
            if spec.target_column in corrupted_df.columns
            else y_true
        )
        try:
            corrupted_metric = target_model.evaluate(corrupted_df, eval_y)
        except Exception:
            # Model catastrophic failure due to corruption
            corrupted_metric = 0.5 if spec.task_type == "classification" else clean_metric * 5.0

        damage = target_model.compute_damage(clean_metric, corrupted_metric)
        execution_time_ms = (time.perf_counter() - t0) * 1000.0

        row = {
            "dataset_name": spec.name,
            "model_id": target_model.model_id,
            "task_type": spec.task_type,
            "metric_name": spec.metric_name,
            "corruption_type": corruption_type,
            "severity": float(severity),
            "clean_metric": float(clean_metric),
            "corrupted_metric": float(corrupted_metric),
            "damage": float(damage),
            "execution_time_ms": float(execution_time_ms),
            **issue_features,
        }
        return row

    def generate_experiment_dataset(
        self,
        n_rows_target: int = 500,
        batch_size: int = 500,
        output_filename: str = "simulation_experiments.parquet",
    ) -> Path:
        """Run batch simulation across all registered datasets and target models."""
        datasets = ["credit_default", "telco_churn", "housing_prices"]
        model_types = ["lightgbm", "random_forest", "linear"]
        corruption_types = [
            "clean",
            "mcar",
            "mar",
            "mnar",
            "mean_shift",
            "variance_scale",
            "unit_error",
            "unseen_categories",
            "concept_drift",
            "label_noise",
            "duplicates",
            "outliers",
            "compound",
        ]
        severities = [0.01, 0.05, 0.10, 0.20, 0.35, 0.50]

        records: list[dict[str, Any]] = []
        seed_counter = self.seed

        for d_name in datasets:
            train_df, test_df, spec, ref_stats = get_dataset(
                name=d_name, n_samples=batch_size * 2, seed=self.seed
            )
            for m_type in model_types:
                target_model = train_target_model(
                    spec=spec,
                    train_df=train_df,
                    ref_stats=ref_stats,
                    model_type=m_type,  # type: ignore[arg-type]
                    seed=self.seed,
                )

                # Generate samples across corruptions
                for c_type in corruption_types:
                    # Skip label noise or concept drift on regression
                    if spec.task_type == "regression" and c_type in (
                        "label_noise",
                        "concept_drift",
                    ):
                        continue

                    for sev in severities:
                        seed_counter += 1
                        rec = self.run_single_simulation(
                            spec=spec,
                            target_model=target_model,
                            clean_ref_df=train_df,
                            test_df=test_df,
                            corruption_type=c_type,
                            severity=sev,
                            seed=seed_counter,
                        )
                        records.append(rec)
                        if len(records) >= n_rows_target:
                            break
                    if len(records) >= n_rows_target:
                        break
                if len(records) >= n_rows_target:
                    break
            if len(records) >= n_rows_target:
                break

        # Save to Parquet
        exp_df = pl.DataFrame(records)
        output_path = self.output_dir / output_filename
        exp_df.write_parquet(output_path)

        # Write manifest
        manifest = {
            "n_records": exp_df.height,
            "datasets": list(exp_df["dataset_name"].unique()),
            "models": list(exp_df["model_id"].unique()),
            "corruption_types": list(exp_df["corruption_type"].unique()),
            "damage_mean": float(exp_df["damage"].mean()),
            "damage_min": float(exp_df["damage"].min()),
            "damage_max": float(exp_df["damage"].max()),
            "columns": exp_df.columns,
            "created_at": time.time(),
        }
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return output_path
