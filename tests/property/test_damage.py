"""Property-based tests for split-conformal intervals and damage predictor monotonicity."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from trustgate.damage.conformal import SplitConformalPredictor
from trustgate.damage.features import DAMAGE_FEATURE_NAMES
from trustgate.damage.predictor import DamagePredictor


@settings(max_examples=50, deadline=None)
@given(
    y_pred=st.floats(min_value=0.0, max_value=1.0),
    alpha=st.floats(min_value=0.01, max_value=0.30),
)
def test_property_conformal_interval_ordering(y_pred: float, alpha: float) -> None:
    """Property: Lower bound <= point estimate <= upper bound, and lower bound >= 0.0."""
    conformal = SplitConformalPredictor(alpha=alpha)
    # Calibrate on dummy residuals
    res = np.linspace(0.01, 0.20, 20)
    conformal.calibration_scores = np.sort(res)
    conformal.quantile_val = float(np.percentile(res, (1.0 - alpha) * 100))
    conformal.is_calibrated = True

    interval = conformal.predict_interval(y_pred)
    assert isinstance(interval, object)
    assert interval.lower_bound <= interval.point_estimate <= interval.upper_bound
    assert interval.lower_bound >= 0.0
    assert interval.interval_width >= 0.0


@settings(max_examples=30, deadline=None)
@given(
    base_val=st.floats(min_value=0.0, max_value=0.2),
    delta=st.floats(min_value=0.01, max_value=0.3),
)
def test_property_monotonic_feature_increasing(base_val: float, delta: float) -> None:
    """Property: Increasing monotonic risk features (null rate, PSI) must never decrease predicted damage."""
    # Create synthetic dataset with positive slope
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 0.5, size=(60, len(DAMAGE_FEATURE_NAMES)))
    y = np.clip(X[:, 0] * 1.5 + X[:, 3] * 1.0, 0.0, 1.0)

    predictor = DamagePredictor(n_estimators=30, seed=42)
    predictor.fit(X, y)

    x1 = np.full(len(DAMAGE_FEATURE_NAMES), base_val)
    x2 = x1.copy()
    x2[0] = base_val + delta  # Increase importance-weighted null rate

    p1 = predictor.predict(x1).predicted_damage
    p2 = predictor.predict(x2).predicted_damage

    # LightGBM monotonic constraint ensures p2 >= p1 (within float precision)
    assert p2 >= p1 - 1e-5
