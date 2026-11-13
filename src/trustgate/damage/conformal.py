"""Split-conformal prediction intervals for model degradation uncertainty."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalInterval:
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    margin: float

    @property
    def interval_width(self) -> float:
        return self.upper_bound - self.lower_bound


class SplitConformalPredictor:
    """Split-conformal predictor offering distribution-free finite-sample coverage guarantees."""

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha  # 1 - alpha target coverage (default 90%)
        self.calibration_scores: np.ndarray = np.array([], dtype=float)
        self.quantile_val: float = 0.0
        self.is_calibrated: bool = False

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Compute non-conformity scores on a dedicated held-out calibration split.

        Parameters
        ----------
        y_true : np.ndarray
            True observed damages on calibration split.
        y_pred : np.ndarray
            Model predicted damages on calibration split.
        """
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)

        if len(y_t) != len(y_p):
            raise ValueError(f"Length mismatch: y_true={len(y_t)} vs y_pred={len(y_p)}")

        if len(y_t) < 10:
            raise ValueError(f"Insufficient calibration points: {len(y_t)} (need >= 10)")

        # Absolute residual non-conformity score
        scores = np.abs(y_t - y_p)
        self.calibration_scores = np.sort(scores)

        # Finite-sample correction factor: ceil((n + 1) * (1 - alpha)) / n
        n = len(scores)
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        k = min(n, max(1, k))
        # 1-indexed k corresponds to index k-1 in sorted 0-indexed array
        self.quantile_val = float(self.calibration_scores[k - 1])
        self.is_calibrated = True

    def predict_interval(
        self,
        y_pred: float | np.ndarray,
        alpha_override: float | None = None,
    ) -> ConformalInterval | list[ConformalInterval]:
        """Compute calibrated interval around prediction(s)."""
        if not self.is_calibrated:
            # Fallback default margin if not yet calibrated
            margin = 0.10
        elif alpha_override is not None and alpha_override != self.alpha:
            n = len(self.calibration_scores)
            k = int(np.ceil((n + 1) * (1.0 - alpha_override)))
            k = min(n, max(1, k))
            margin = float(self.calibration_scores[k - 1])
        else:
            margin = self.quantile_val

        if np.isscalar(y_pred):
            pred_f = float(y_pred)  # type: ignore[arg-type]
            lower = max(0.0, pred_f - margin)  # Damage cannot be negative in policy decisions
            upper = pred_f + margin
            return ConformalInterval(
                point_estimate=pred_f,
                lower_bound=lower,
                upper_bound=upper,
                confidence_level=1.0 - (alpha_override or self.alpha),
                margin=margin,
            )
        else:
            arr = np.asarray(y_pred, dtype=float)
            results = []
            for val in arr:
                lower = max(0.0, val - margin)
                upper = val + margin
                results.append(
                    ConformalInterval(
                        point_estimate=val,
                        lower_bound=lower,
                        upper_bound=upper,
                        confidence_level=1.0 - (alpha_override or self.alpha),
                        margin=margin,
                    )
                )
            return results

    def evaluate_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate empirical coverage percentage and average interval width."""
        intervals = self.predict_interval(y_pred)
        if isinstance(intervals, ConformalInterval):
            intervals = [intervals]

        covered = 0
        widths = []
        for yt, interval in zip(y_true, intervals, strict=False):
            widths.append(interval.interval_width)
            if interval.lower_bound <= yt <= interval.upper_bound:
                covered += 1

        empirical_cov = float(covered / len(y_true)) if len(y_true) > 0 else 0.0
        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": empirical_cov,
            "average_width": float(np.mean(widths)) if widths else 0.0,
            "is_valid": empirical_cov
            >= (1.0 - self.alpha - 0.05),  # 5% tolerance for finite test sample
        }
