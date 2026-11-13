"""Unit tests for Phase 5 policy evaluation engine, risk scores, hard blocks, and policy simulator."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trustgate.damage.features import DAMAGE_FEATURE_NAMES
from trustgate.damage.predictor import DamagePredictor
from trustgate.pii.detectors import PIICategory, PIIFinding, PIISeverity
from trustgate.pii.scanner import PIIScanReport
from trustgate.policy.engine import Decision, PolicyEngine
from trustgate.policy.simulator import PolicySimulator
from trustgate.simulator.datasets import get_dataset
from trustgate.simulator.models import train_target_model


def test_hard_block_on_critical_pii() -> None:
    """Verify unmasked direct PII triggers immediate QUARANTINE and 100 Risk Score."""
    train_df, test_df, spec, ref_stats = get_dataset("credit_default", n_samples=200, seed=42)
    model = train_target_model(spec, train_df, ref_stats, model_type="lightgbm", seed=42)

    X = np.random.uniform(0, 0.2, size=(30, len(DAMAGE_FEATURE_NAMES)))
    y = np.random.uniform(0.01, 0.2, size=30)
    predictor = DamagePredictor(n_estimators=20, seed=42).fit(X, y)

    # Synthetic critical PII report
    finding = PIIFinding(
        entity_type="indian_aadhaar",
        raw_value="9999 1234 5678",
        confidence=0.99,
        severity=PIISeverity.CRITICAL,
        category=PIICategory.DIRECT_IDENTIFIER,
    )
    pii_report = PIIScanReport(
        contains_unmasked_direct_pii=True,
        has_critical_pii=True,
        total_findings_count=1,
        pii_columns={"col": [finding]},
        findings_by_severity={"CRITICAL": 1},
    )

    engine = PolicyEngine()
    record, out_df = engine.evaluate(
        batch_id="batch_pii_1",
        df=test_df,
        spec=spec,
        target_model=model,
        ref_df=train_df,
        ref_stats=ref_stats,
        damage_predictor=predictor,
        pii_report=pii_report,
    )

    assert record.decision == Decision.QUARANTINE
    assert record.risk_score == 100.0
    assert record.hard_block_triggered
    assert out_df is None


def test_policy_decision_clean_batch_approve(tmp_path: Path) -> None:
    """Verify clean batch receives APPROVE or REPAIR_THEN_APPROVE."""
    from trustgate.simulator.runner import ExperimentRunner

    # Train predictor on realistic simulation dataset
    runner = ExperimentRunner(output_dir=str(tmp_path), seed=42)
    p = runner.generate_experiment_dataset(n_rows_target=25, batch_size=100)
    exp_df = pl.read_parquet(p)
    X = exp_df.select(DAMAGE_FEATURE_NAMES).to_numpy()
    y = exp_df["damage"].to_numpy()
    predictor = DamagePredictor(n_estimators=30, seed=42).fit(X, y)

    train_df, test_df, spec, ref_stats = get_dataset("credit_default", n_samples=200, seed=42)
    model = train_target_model(spec, train_df, ref_stats, model_type="lightgbm", seed=42)

    engine = PolicyEngine()
    record, out_df = engine.evaluate(
        batch_id="batch_clean_1",
        df=test_df,
        spec=spec,
        target_model=model,
        ref_df=train_df,
        ref_stats=ref_stats,
        damage_predictor=predictor,
    )

    assert record.decision in (Decision.APPROVE, Decision.REPAIR_THEN_APPROVE)
    assert record.damage_upper <= 0.05
    assert not record.hard_block_triggered
    assert record.risk_score < 30.0
    assert out_df is not None


def test_policy_simulator_sweep() -> None:
    """Verify PolicySimulator generates trade-off points across threshold sweeps."""
    damages = [0.01, 0.02, 0.04, 0.06, 0.12, 0.25, 0.40]
    exp_df = pl.DataFrame({"damage": damages})

    sim = PolicySimulator(exp_df)
    points = sim.simulate_threshold_sweep(approve_thresholds=[0.03, 0.10, 0.20])

    assert len(points) == 3
    # At tau=0.03: 0.01 and 0.02 are approved (2/7)
    assert points[0].approved_count == 2
    assert points[0].approval_rate == pytest.approx(2 / 7, abs=1e-3)
    # Stricter threshold means lower or equal average approved damage
    assert points[0].avg_approved_damage <= points[1].avg_approved_damage
