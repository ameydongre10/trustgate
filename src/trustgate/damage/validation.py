"""Rigorous validation protocols: Random split, LOCO, LODO, and LOMO cross-validation."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from trustgate.damage.baselines import (
    LinearDamageBaseline,
    MaxMetricDamageBaseline,
    RuleCountDamageBaseline,
)
from trustgate.damage.features import DAMAGE_FEATURE_NAMES
from trustgate.damage.predictor import DamagePredictor


@dataclass
class ProtocolMetrics:
    protocol_name: str
    rmse: float
    mae: float
    spearman_corr: float
    conformal_coverage: float
    baseline_linear_rmse: float
    baseline_rule_rmse: float
    baseline_max_metric_rmse: float
    details: dict[str, Any]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    predictor: DamagePredictor,
    baselines: dict[str, Any],
    X_test: np.ndarray,
    protocol_name: str,
    details: dict[str, Any] | None = None,
) -> ProtocolMetrics:
    """Compute standard metrics comparing damage predictor against baselines."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # Spearman rank correlation
    if len(np.unique(y_pred)) > 1 and len(np.unique(y_true)) > 1:
        spearman_res = stats.spearmanr(y_true, y_pred)
        spearman_corr = float(spearman_res.statistic)
    else:
        spearman_corr = 0.0

    # Conformal empirical coverage
    cov_eval = predictor.conformal.evaluate_coverage(y_true, y_pred)
    coverage = cov_eval["empirical_coverage"]

    # Baseline RMSEs
    lin_pred = baselines["linear"].predict(X_test)
    rule_pred = baselines["rule"].predict(X_test)
    max_pred = baselines["max_metric"].predict(X_test)

    lin_rmse = float(np.sqrt(mean_squared_error(y_true, lin_pred)))
    rule_rmse = float(np.sqrt(mean_squared_error(y_true, rule_pred)))
    max_rmse = float(np.sqrt(mean_squared_error(y_true, max_pred)))

    return ProtocolMetrics(
        protocol_name=protocol_name,
        rmse=rmse,
        mae=mae,
        spearman_corr=spearman_corr,
        conformal_coverage=coverage,
        baseline_linear_rmse=lin_rmse,
        baseline_rule_rmse=rule_rmse,
        baseline_max_metric_rmse=max_rmse,
        details=details or {},
    )


class ValidationSuite:
    """Evaluates DamagePredictor across the 4 strict generalization protocols."""

    def __init__(self, data_df: pl.DataFrame, seed: int = 42) -> None:
        self.data_df = data_df
        self.seed = seed
        self.feature_cols = [c for c in DAMAGE_FEATURE_NAMES if c in data_df.columns]

    def _prepare_matrices(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df.select(self.feature_cols).to_numpy().astype(np.float32)
        y = df["damage"].to_numpy().astype(np.float32)
        return X, y

    def _fit_models(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[DamagePredictor, dict[str, Any]]:
        # Calibrate on last 20% of train
        n_calib = max(10, int(len(X_train) * 0.2))
        X_tr, y_tr = X_train[:-n_calib], y_train[:-n_calib]
        X_cal, y_cal = X_train[-n_calib:], y_train[-n_calib:]

        predictor = DamagePredictor(seed=self.seed)
        predictor.fit(X_tr, y_tr, X_calib=X_cal, y_calib=y_cal)

        baselines = {
            "linear": LinearDamageBaseline().fit(X_tr, y_tr),
            "rule": RuleCountDamageBaseline().fit(X_tr, y_tr),
            "max_metric": MaxMetricDamageBaseline().fit(X_tr, y_tr),
        }
        return predictor, baselines

    def run_random_split(self, test_size: float = 0.25) -> ProtocolMetrics:
        """Protocol (a): Standard in-distribution random split."""
        n_total = self.data_df.height
        shuffled = self.data_df.sample(fraction=1.0, shuffle=True, seed=self.seed)
        n_train = int(n_total * (1.0 - test_size))

        train_df = shuffled.slice(0, n_train)
        test_df = shuffled.slice(n_train, n_total - n_train)

        X_train, y_train = self._prepare_matrices(train_df)
        X_test, y_test = self._prepare_matrices(test_df)

        predictor, baselines = self._fit_models(X_train, y_train)
        y_pred = predictor.model.predict(X_test)

        return compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            predictor=predictor,
            baselines=baselines,
            X_test=X_test,
            protocol_name="random_split",
        )

    def run_leave_one_corruption_out(self) -> dict[str, ProtocolMetrics]:
        """Protocol (b): Leave-One-Corruption-Type-Out (LOCO)."""
        results: dict[str, ProtocolMetrics] = {}
        unique_corruptions = [c for c in self.data_df["corruption_type"].unique() if c != "clean"]

        for held_out in unique_corruptions:
            train_df = self.data_df.filter(pl.col("corruption_type") != held_out)
            test_df = self.data_df.filter(pl.col("corruption_type") == held_out)

            if test_df.height < 5 or train_df.height < 20:
                continue

            X_train, y_train = self._prepare_matrices(train_df)
            X_test, y_test = self._prepare_matrices(test_df)

            predictor, baselines = self._fit_models(X_train, y_train)
            y_pred = predictor.model.predict(X_test)

            results[held_out] = compute_metrics(
                y_true=y_test,
                y_pred=y_pred,
                predictor=predictor,
                baselines=baselines,
                X_test=X_test,
                protocol_name=f"LOCO_{held_out}",
                details={"held_out_corruption": held_out, "n_test": test_df.height},
            )

        return results

    def run_leave_one_dataset_out(self) -> dict[str, ProtocolMetrics]:
        """Protocol (c): Leave-One-Dataset-Out (LODO)."""
        results: dict[str, ProtocolMetrics] = {}
        unique_datasets = list(self.data_df["dataset_name"].unique())

        for held_out in unique_datasets:
            train_df = self.data_df.filter(pl.col("dataset_name") != held_out)
            test_df = self.data_df.filter(pl.col("dataset_name") == held_out)

            if test_df.height < 5 or train_df.height < 20:
                continue

            X_train, y_train = self._prepare_matrices(train_df)
            X_test, y_test = self._prepare_matrices(test_df)

            predictor, baselines = self._fit_models(X_train, y_train)
            y_pred = predictor.model.predict(X_test)

            results[held_out] = compute_metrics(
                y_true=y_test,
                y_pred=y_pred,
                predictor=predictor,
                baselines=baselines,
                X_test=X_test,
                protocol_name=f"LODO_{held_out}",
                details={"held_out_dataset": held_out, "n_test": test_df.height},
            )

        return results

    def run_leave_one_model_out(self) -> dict[str, ProtocolMetrics]:
        """Protocol (d): Leave-One-Target-Model-Out (LOMO)."""
        results: dict[str, ProtocolMetrics] = {}
        unique_models = list(self.data_df["model_id"].unique())

        for held_out in unique_models:
            train_df = self.data_df.filter(pl.col("model_id") != held_out)
            test_df = self.data_df.filter(pl.col("model_id") == held_out)

            if test_df.height < 5 or train_df.height < 20:
                continue

            X_train, y_train = self._prepare_matrices(train_df)
            X_test, y_test = self._prepare_matrices(test_df)

            predictor, baselines = self._fit_models(X_train, y_train)
            y_pred = predictor.model.predict(X_test)

            results[held_out] = compute_metrics(
                y_true=y_test,
                y_pred=y_pred,
                predictor=predictor,
                baselines=baselines,
                X_test=X_test,
                protocol_name=f"LOMO_{held_out}",
                details={"held_out_model": held_out, "n_test": test_df.height},
            )

        return results
