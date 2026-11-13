"""Out-of-Distribution (OOD) Guard detecting issue profiles far from training support."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

from trustgate.damage.features import DAMAGE_FEATURE_NAMES, FEATURE_HUMAN_LABELS


@dataclass
class OODCheckResult:
    is_ood: bool
    confidence_score: float
    anomaly_score: float
    reason: str
    extreme_features: list[str]


class OODGuard:
    """Detects when an incoming batch's issue profile is out-of-distribution."""

    def __init__(self, contamination: float = 0.02, seed: int = 42) -> None:
        self.contamination = contamination
        self.seed = seed
        self.iso_forest: IsolationForest | None = None
        self.mean_vector: np.ndarray | None = None
        self.inv_cov_matrix: np.ndarray | None = None
        self.mahalanobis_threshold: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "OODGuard":
        """Fit Isolation Forest and robust Mahalanobis distance on clean/corrupted training profiles."""
        X_mat = np.asarray(X, dtype=np.float64)
        if len(X_mat) < 10:
            return self

        # 1. Isolation Forest
        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.seed,
            n_estimators=100,
        )
        self.iso_forest.fit(X_mat)

        # 2. Regularized Mahalanobis parameters
        self.mean_vector = np.mean(X_mat, axis=0)
        cov = np.cov(X_mat, rowvar=False)
        # Regularize covariance to guarantee invertibility
        cov_reg = cov + np.eye(X_mat.shape[1]) * 1e-4
        self.inv_cov_matrix = np.linalg.pinv(cov_reg)

        # Compute training distances to set threshold at 99th percentile
        diffs = X_mat - self.mean_vector
        dists = np.sqrt(np.sum((diffs @ self.inv_cov_matrix) * diffs, axis=1))
        self.mahalanobis_threshold = float(np.percentile(dists, 99.0))

        self.is_fitted = True
        return self

    def check(self, x: np.ndarray) -> OODCheckResult:
        """Evaluate whether a single feature vector is out-of-distribution."""
        if not self.is_fitted or self.iso_forest is None or self.mean_vector is None:
            # Not fitted: fail-safe assume in-distribution with neutral confidence
            return OODCheckResult(
                is_ood=False,
                confidence_score=0.80,
                anomaly_score=0.0,
                reason="OOD guard not yet fitted",
                extreme_features=[],
            )

        x_vec = np.asarray(x, dtype=np.float64).reshape(1, -1)

        # Isolation forest score (lower is more anomalous; negative means outlier)
        if_score = float(self.iso_forest.score_samples(x_vec)[0])
        is_if_outlier = bool(self.iso_forest.predict(x_vec)[0] == -1)

        # Mahalanobis distance
        diff = x_vec[0] - self.mean_vector
        m_dist = float(np.sqrt(diff @ self.inv_cov_matrix @ diff))
        is_m_outlier = m_dist > self.mahalanobis_threshold

        is_ood = is_if_outlier or is_m_outlier

        # Identify features deviating significantly from training mean
        std_est = np.sqrt(np.diag(np.linalg.pinv(self.inv_cov_matrix)))
        z_scores = np.abs(diff) / np.maximum(std_est, 1e-4)
        extreme_indices = np.where(z_scores > 3.5)[0]
        extreme_features = [
            FEATURE_HUMAN_LABELS.get(DAMAGE_FEATURE_NAMES[idx], DAMAGE_FEATURE_NAMES[idx])
            for idx in extreme_indices
            if idx < len(DAMAGE_FEATURE_NAMES)
        ]

        # Confidence drops smoothly as anomaly increases
        confidence = float(
            np.clip(1.0 - (m_dist / max(1.0, self.mahalanobis_threshold * 2.0)), 0.05, 0.99)
        )
        if is_ood:
            confidence = min(confidence, 0.45)  # Force confidence < 0.50 on OOD

        reason = (
            f"Issue profile deviates from training support (Mahalanobis={m_dist:.1f}, IF={if_score:.2f})"
            if is_ood
            else "Issue profile lies within known training support"
        )

        return OODCheckResult(
            is_ood=is_ood,
            confidence_score=confidence,
            anomaly_score=m_dist,
            reason=reason,
            extreme_features=extreme_features,
        )
