"""Baseline damage predictors: linear model, rule-count heuristic, and max-single-metric heuristic."""

from typing import cast

import numpy as np
from sklearn.linear_model import Ridge


class LinearDamageBaseline:
    """Baseline 1: Ridge regression with non-negative projection."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.model = Ridge(alpha=alpha, positive=True)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearDamageBaseline":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(X), dtype=np.float32)
        preds = self.model.predict(X)
        return cast(np.ndarray, np.asarray(np.maximum(0.0, preds), dtype=np.float32))


class RuleCountDamageBaseline:
    """Baseline 2: Rule-count heuristic counting threshold breaches."""

    def __init__(self) -> None:
        self.scale_factor: float = 0.25

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RuleCountDamageBaseline":
        rule_violations = self._count_violations(X)
        if np.sum(rule_violations) > 0:
            mean_y = float(np.mean(y))
            mean_v = float(np.mean(rule_violations))
            self.scale_factor = float(mean_y / max(0.1, mean_v))
        return self

    def _count_violations(self, X: np.ndarray) -> np.ndarray:
        violations = (
            (X[:, 0] > 0.05).astype(np.float32)
            + (X[:, 1] > 0.15).astype(np.float32)
            + (X[:, 3] > 0.10).astype(np.float32)
            + (X[:, 4] > 0.25).astype(np.float32)
            + (X[:, 7] > 0.03).astype(np.float32)
            + (X[:, 8] > 0.05).astype(np.float32)
        )
        return cast(np.ndarray, np.asarray(violations, dtype=np.float32))

    def predict(self, X: np.ndarray) -> np.ndarray:
        violations = self._count_violations(X)
        return cast(
            np.ndarray,
            np.asarray(np.clip(violations * self.scale_factor, 0.0, 1.0), dtype=np.float32),
        )


class MaxMetricDamageBaseline:
    """Baseline 3: 'Max single metric' heuristic taking the worst observed violation."""

    def __init__(self) -> None:
        self.multiplier: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MaxMetricDamageBaseline":
        max_metrics = self._extract_max(X)
        mean_m = float(np.mean(max_metrics))
        if mean_m > 0:
            self.multiplier = float(np.mean(y) / mean_m)
        return self

    def _extract_max(self, X: np.ndarray) -> np.ndarray:
        null_part = X[:, 0]
        psi_part = X[:, 3] * 0.5
        outlier_part = X[:, 7] * 2.0
        return cast(
            np.ndarray,
            np.asarray(
                np.maximum(np.maximum(null_part, psi_part), outlier_part), dtype=np.float32
            ),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        max_vals = self._extract_max(X)
        return cast(
            np.ndarray,
            np.asarray(np.clip(max_vals * self.multiplier, 0.0, 1.0), dtype=np.float32),
        )
