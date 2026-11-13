"""Monotonic LightGBM damage predictor with split-conformal intervals and TreeSHAP."""

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np

from trustgate.damage.conformal import ConformalInterval, SplitConformalPredictor
from trustgate.damage.features import (
    DAMAGE_FEATURE_NAMES,
    DAMAGE_MONOTONIC_CONSTRAINTS,
    FEATURE_HUMAN_LABELS,
    build_feature_vector,
)
from trustgate.damage.ood_guard import OODCheckResult, OODGuard


@dataclass
class DamagePrediction:
    predicted_damage: float
    damage_lower: float
    damage_upper: float
    confidence: float
    is_ood: bool
    ood_score: float
    conformal_margin: float
    shap_values: dict[str, float]
    top_factors: list[tuple[str, float]]
    summary: str


class DamagePredictor:
    """Model-aware damage predictor with monotonic constraints, conformal bounds, and TreeSHAP."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        alpha: float = 0.10,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.seed = seed

        # Core LightGBM regressor with monotonic constraints
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            monotone_constraints=DAMAGE_MONOTONIC_CONSTRAINTS,
            random_state=self.seed,
            min_child_samples=5,
            verbose=-1,
        )

        self.conformal = SplitConformalPredictor(alpha=self.alpha)
        self.ood_guard = OODGuard(seed=self.seed)
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray | None = None,
        y_calib: np.ndarray | None = None,
    ) -> "DamagePredictor":
        """Fit monotonic LightGBM regressor, calibrate conformal predictor, and fit OOD guard."""
        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.float32)

        # 1. Fit monotonic LightGBM model
        self.model.fit(X_tr, y_tr)

        # 2. Calibrate conformal intervals
        if X_calib is not None and y_calib is not None and len(X_calib) >= 10:
            calib_preds = self.model.predict(np.asarray(X_calib, dtype=np.float32))
            self.conformal.calibrate(y_true=y_calib, y_pred=calib_preds)
        else:
            # Self-calibrate on holdout slice if calibration set not passed separately
            split_idx = int(len(X_tr) * 0.8)
            if split_idx >= 10:
                calib_preds = self.model.predict(X_tr[split_idx:])
                self.conformal.calibrate(y_true=y_tr[split_idx:], y_pred=calib_preds)

        # 3. Fit OOD guard
        self.ood_guard.fit(X_tr)

        self.is_fitted = True
        return self

    def predict(
        self,
        issue_input: dict[str, Any] | np.ndarray,
        alpha_override: float | None = None,
    ) -> DamagePrediction:
        """Predict expected downstream metric damage with conformal interval and TreeSHAP."""
        if not self.is_fitted:
            raise RuntimeError("DamagePredictor must be fitted before predict() is called.")

        if isinstance(issue_input, dict):
            feat_vec = build_feature_vector(issue_input)
        else:
            feat_vec = np.asarray(issue_input, dtype=np.float32).ravel()

        X_in = feat_vec.reshape(1, -1)

        # 1. Point prediction
        raw_pred = float(self.model.predict(X_in)[0])
        # Damage is non-negative for policy safety (clipped)
        point_pred = max(0.0, raw_pred)

        # 2. Conformal prediction interval
        conformal_res = self.conformal.predict_interval(point_pred, alpha_override=alpha_override)
        if isinstance(conformal_res, list):
            interval: ConformalInterval = conformal_res[0]
        else:
            interval = conformal_res

        # 3. OOD verification
        ood_res: OODCheckResult = self.ood_guard.check(feat_vec)

        # 4. Native TreeSHAP computation via LightGBM (ADR-006)
        # booster.predict with pred_contrib=True yields [phi_1, phi_2, ..., phi_k, expected_value]
        booster = self.model.booster_
        contribs = booster.predict(X_in, pred_contrib=True)[0]
        feature_shap = contribs[:-1]
        float(contribs[-1])

        shap_dict = {
            DAMAGE_FEATURE_NAMES[i]: float(feature_shap[i])
            for i in range(len(DAMAGE_FEATURE_NAMES))
        }

        # Rank top drivers of damage (positive SHAP contributions)
        top_factors = []
        for feat_name, shap_val in sorted(
            shap_dict.items(), key=lambda item: item[1], reverse=True
        ):
            if shap_val > 0.001:
                friendly_name = FEATURE_HUMAN_LABELS.get(feat_name, feat_name)
                top_factors.append((friendly_name, round(shap_val, 4)))

        # Format concise human-readable explanation
        if ood_res.is_ood:
            summary = (
                f"Predicted damage {point_pred * 100:.1f}% [upper bound {interval.upper_bound * 100:.1f}%]. "
                f"CAUTION: Batch issue profile is out-of-distribution ({ood_res.reason})."
            )
        elif not top_factors:
            summary = (
                f"Predicted negligible damage {point_pred * 100:.1f}% [upper bound {interval.upper_bound * 100:.1f}%]. "
                "No critical degradation factors identified."
            )
        else:
            driver_str = ", ".join([f"{name} (+{val * 100:.1f}%)" for name, val in top_factors[:2]])
            summary = (
                f"Predicted damage {point_pred * 100:.1f}% [upper bound {interval.upper_bound * 100:.1f}%]. "
                f"Primary drivers: {driver_str}."
            )

        return DamagePrediction(
            predicted_damage=point_pred,
            damage_lower=interval.lower_bound,
            damage_upper=interval.upper_bound,
            confidence=ood_res.confidence_score,
            is_ood=ood_res.is_ood,
            ood_score=ood_res.anomaly_score,
            conformal_margin=interval.margin,
            shap_values=shap_dict,
            top_factors=top_factors[:5],
            summary=summary,
        )
