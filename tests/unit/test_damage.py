"""Unit tests for Phase 4 damage predictor, conformal intervals, OOD guard, baselines, and registry."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trustgate.damage.baselines import (
    LinearDamageBaseline,
    MaxMetricDamageBaseline,
    RuleCountDamageBaseline,
)
from trustgate.damage.conformal import SplitConformalPredictor
from trustgate.damage.features import (
    DAMAGE_FEATURE_NAMES,
    build_feature_vector,
)
from trustgate.damage.ood_guard import OODGuard
from trustgate.damage.predictor import DamagePrediction, DamagePredictor
from trustgate.damage.registry import ModelRegistry
from trustgate.damage.validation import ValidationSuite


def test_feature_vector_builder() -> None:
    """Verify issue metrics dict converts to exact 1D numpy array."""
    metrics = {
        "importance_weighted_null_rate": 0.15,
        "max_null_rate": 0.40,
        "importance_weighted_psi": 0.22,
        "outlier_rate": 0.05,
        "batch_size": 1000,
    }
    vec = build_feature_vector(metrics)
    assert len(vec) == len(DAMAGE_FEATURE_NAMES)
    assert vec[0] == pytest.approx(0.15, abs=1e-4)
    assert vec[1] == pytest.approx(0.40, abs=1e-4)
    assert vec[3] == pytest.approx(0.22, abs=1e-4)


def test_split_conformal_predictor() -> None:
    """Verify conformal predictor calibration, intervals, and coverage guarantees."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.uniform(0.0, 0.5, size=n)
    # Predictions with noise
    y_pred = np.clip(y_true + rng.normal(0, 0.05, size=n), 0.0, 1.0)

    conformal = SplitConformalPredictor(alpha=0.10)
    conformal.calibrate(y_true=y_true, y_pred=y_pred)
    assert conformal.is_calibrated
    assert conformal.quantile_val > 0.0

    # Single prediction interval
    interval = conformal.predict_interval(0.25)
    assert isinstance(interval, object)
    assert interval.lower_bound <= 0.25 <= interval.upper_bound
    assert interval.lower_bound >= 0.0  # Clipped

    # Empirical coverage on test set
    y_test_true = rng.uniform(0.0, 0.5, size=100)
    y_test_pred = np.clip(y_test_true + rng.normal(0, 0.05, size=100), 0.0, 1.0)
    cov_report = conformal.evaluate_coverage(y_test_true, y_test_pred)
    assert cov_report["empirical_coverage"] >= 0.80


def test_ood_guard() -> None:
    """Verify OOD guard flags points outside training support and drops confidence."""
    rng = np.random.default_rng(42)
    # Training points centered near 0.1
    X_train = rng.normal(0.1, 0.02, size=(100, len(DAMAGE_FEATURE_NAMES)))

    guard = OODGuard(seed=42).fit(X_train)
    assert guard.is_fitted

    # In-distribution point
    in_dist = np.full(len(DAMAGE_FEATURE_NAMES), 0.1)
    res_in = guard.check(in_dist)
    assert not res_in.is_ood
    assert res_in.confidence_score >= 0.50

    # Wildly out-of-distribution point (e.g. extreme values of 50.0)
    out_dist = np.full(len(DAMAGE_FEATURE_NAMES), 50.0)
    res_out = guard.check(out_dist)
    assert res_out.is_ood
    assert res_out.confidence_score < 0.50
    assert len(res_out.extreme_features) > 0


def test_baselines_fit_and_predict() -> None:
    """Verify linear, rule-count, and max-metric damage baselines."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 0.3, size=(50, len(DAMAGE_FEATURE_NAMES)))
    y = np.clip(X[:, 0] * 1.5 + X[:, 3] * 0.8 + rng.normal(0, 0.02, size=50), 0.0, 1.0)

    lin = LinearDamageBaseline().fit(X, y)
    rule = RuleCountDamageBaseline().fit(X, y)
    max_m = MaxMetricDamageBaseline().fit(X, y)

    for b in [lin, rule, max_m]:
        preds = b.predict(X[:5])
        assert len(preds) == 5
        assert np.all(preds >= 0.0)


def test_damage_predictor_fit_predict_and_shap() -> None:
    """Verify monotonic LightGBM DamagePredictor training, conformal bounds, and TreeSHAP."""
    rng = np.random.default_rng(42)
    n = 120
    X = rng.uniform(0.0, 0.4, size=(n, len(DAMAGE_FEATURE_NAMES)))
    # True damage increases with null rate and PSI
    y = np.clip(X[:, 0] * 1.2 + X[:, 3] * 0.9 + rng.normal(0, 0.01, size=n), 0.0, 1.0)

    predictor = DamagePredictor(n_estimators=50, seed=42)
    predictor.fit(X, y)
    assert predictor.is_fitted

    # Single prediction
    pred_res = predictor.predict(X[0])
    assert isinstance(pred_res, DamagePrediction)
    assert pred_res.damage_lower <= pred_res.predicted_damage <= pred_res.damage_upper
    assert pred_res.damage_lower >= 0.0
    assert len(pred_res.shap_values) == len(DAMAGE_FEATURE_NAMES)
    assert isinstance(pred_res.summary, str)

    # Monotonicity test: increasing null rate (index 0) must NOT decrease predicted damage
    x_base = np.zeros(len(DAMAGE_FEATURE_NAMES))
    x_higher_null = x_base.copy()
    x_higher_null[0] = 0.30

    p_base = predictor.predict(x_base).predicted_damage
    p_high = predictor.predict(x_higher_null).predicted_damage
    assert p_high >= p_base - 1e-6


def test_validation_suite_and_registry(tmp_path: Path) -> None:
    """Verify 4-protocol ValidationSuite and hash-pinned ModelRegistry with Model Card."""
    rng = np.random.default_rng(42)
    n = 80
    records = []
    for i in range(n):
        c_type = "mcar" if i < 30 else ("mean_shift" if i < 60 else "outliers")
        d_name = "credit_default" if i < 40 else "telco_churn"
        m_id = "model_a" if i < 40 else "model_b"

        rec = {
            "dataset_name": d_name,
            "model_id": m_id,
            "corruption_type": c_type,
            "damage": float(rng.uniform(0.01, 0.5)),
            "importance_weighted_null_rate": float(rng.uniform(0, 0.3)),
            "max_null_rate": float(rng.uniform(0, 0.5)),
            "mean_null_rate": float(rng.uniform(0, 0.2)),
            "importance_weighted_psi": float(rng.uniform(0, 0.3)),
            "max_psi": float(rng.uniform(0, 0.4)),
            "mean_psi": float(rng.uniform(0, 0.2)),
            "max_ks_stat": float(rng.uniform(0, 0.4)),
            "outlier_rate": float(rng.uniform(0, 0.1)),
            "duplicate_rate": float(rng.uniform(0, 0.05)),
            "class_balance_ratio": 0.5,
            "schema_columns_missing": 0.0,
            "schema_columns_extra": 0.0,
            "batch_size": 500.0,
            "n_columns": 10.0,
        }
        records.append(rec)

    df = pl.DataFrame(records)
    suite = ValidationSuite(df, seed=42)

    # 1. Random split
    random_metrics = suite.run_random_split(test_size=0.25)
    assert random_metrics.rmse >= 0.0
    assert random_metrics.protocol_name == "random_split"

    # 2. LOCO
    loco_results = suite.run_leave_one_corruption_out()
    assert len(loco_results) > 0

    # 3. LODO
    lodo_results = suite.run_leave_one_dataset_out()
    assert len(lodo_results) > 0

    # 4. Model Registry
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    predictor = DamagePredictor(n_estimators=30, seed=42)
    X = df.select(DAMAGE_FEATURE_NAMES).to_numpy()
    y = df["damage"].to_numpy()
    predictor.fit(X, y)

    artifact_path = registry.register(predictor, version="1.0.0")
    assert artifact_path.exists()

    loaded = registry.load(version="1.0.0", verify_hash=True)
    assert loaded.is_fitted

    # Tampering test: modify 1 byte of artifact file -> load must raise ValueError
    with open(artifact_path, "ab") as f:
        f.write(b"tamper_byte")

    with pytest.raises(ValueError, match="Integrity check failed"):
        registry.load(version="1.0.0", verify_hash=True)

    # Generate Model Card
    card_path = registry.generate_model_card(
        random_split_metrics=random_metrics,
        loco_metrics=loco_results,
        lodo_metrics=lodo_results,
        output_path=str(tmp_path / "docs" / "model_card.md"),
    )
    assert card_path.exists()
    assert "Mitchell et al." in card_path.read_text(
        encoding="utf-8"
    ) or "Model Card" in card_path.read_text(encoding="utf-8")
