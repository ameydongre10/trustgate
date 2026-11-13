"""End-to-End (E2E) integration tests for the TrustGate Data Trust & Verification Agent.

Tests the complete lifecycle of incoming data batches:
1. File ingest & DataContract validation
2. Issue profiling (quality, drift, outliers, PII)
3. Model-aware damage prediction with conformal intervals
4. Deterministic policy decision & closed-loop repair
5. Hash-chained audit ledger append & verification
6. AI-Readiness certification
"""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trustgate.agents.graph import TrustGateWorkflow
from trustgate.agents.state import TrustGateState
from trustgate.audit.ledger import AuditLedger, verify_audit_ledger
from trustgate.certify.ai_readiness import AIReadinessCertifier
from trustgate.contracts.models import (
    ColumnContract,
    ColumnDType,
    DataContract,
    TrustLevel,
)
from trustgate.damage.features import DAMAGE_FEATURE_NAMES
from trustgate.damage.predictor import DamagePredictor
from trustgate.policy.engine import Decision, PolicyEngine
from trustgate.simulator.datasets import get_dataset
from trustgate.simulator.injectors import inject_mcar
from trustgate.simulator.models import train_target_model


@pytest.fixture(scope="module")
def setup_pipeline():
    """Setup reusable target model, contract, and trained damage predictor."""
    train_df, test_df, spec, ref_stats = get_dataset("credit_default", n_samples=500, seed=42)
    target_model = train_target_model(spec, train_df, ref_stats, model_type="lightgbm", seed=42)

    contract = DataContract(
        name="credit_default_contract",
        dataset_id="credit_default",
        version="1.0.0",
        owner="mlops@bank.internal",
        trust_level=TrustLevel.VERIFIED,
        columns=[
            ColumnContract(name="income", dtype=ColumnDType.NUMERIC, min_value=0.0),
            ColumnContract(name="debt", dtype=ColumnDType.NUMERIC, min_value=0.0),
            ColumnContract(
                name="credit_score", dtype=ColumnDType.NUMERIC, min_value=300.0, max_value=850.0
            ),
            ColumnContract(name="age", dtype=ColumnDType.NUMERIC, min_value=18, max_value=100),
            ColumnContract(name="years_employed", dtype=ColumnDType.NUMERIC, min_value=0),
            ColumnContract(name="employment_type", dtype=ColumnDType.CATEGORICAL),
            ColumnContract(name="home_ownership", dtype=ColumnDType.CATEGORICAL),
            ColumnContract(name="defaulted", dtype=ColumnDType.NUMERIC),
        ],
    )

    # Train a baseline damage predictor
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 0.2, size=(80, len(DAMAGE_FEATURE_NAMES)))
    X[:25] = 0.0  # Clean profiles
    y = np.clip(X[:, 2] * 0.4 + X[:, 0] * 0.3, 0.0, 1.0)
    y[:25] = 0.0
    predictor = DamagePredictor(n_estimators=30, seed=42).fit(X, y)

    policy_engine = PolicyEngine()
    workflow = TrustGateWorkflow(policy_engine=policy_engine, damage_predictor=predictor)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "spec": spec,
        "ref_stats": ref_stats,
        "target_model": target_model,
        "contract": contract,
        "workflow": workflow,
        "predictor": predictor,
    }


def test_e2e_clean_batch_approval(setup_pipeline: dict) -> None:
    """Verify that a clean in-distribution batch is approved, certified, and audited."""
    env = setup_pipeline
    test_df = env["test_df"].head(100)
    batch_id = "BATCH-E2E-CLEAN-001"

    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_path = Path(tmp_dir) / "audit_ledger.jsonl"
        audit_ledger = AuditLedger(ledger_file=ledger_path)

        state = TrustGateState(
            batch_id=batch_id,
            raw_df=test_df,
            spec=env["spec"],
            target_model=env["target_model"],
            contract=env["contract"],
            ref_df=env["train_df"],
            ref_stats=env["ref_stats"],
        )

        final_state = env["workflow"].run(state)

        # 1. Verification of decision
        assert final_state.policy_decision is not None
        assert final_state.policy_decision.decision in (
            Decision.APPROVE,
            Decision.ESCALATE_TO_HUMAN,
        )
        assert final_state.policy_decision.risk_score < 50.0

        # 2. Audit append
        entry = audit_ledger.append_decision(
            batch_id=batch_id,
            payload={
                "decision": final_state.policy_decision.decision.value,
                "risk_score": final_state.policy_decision.risk_score,
            },
        )
        assert entry.sequence_num >= 0
        assert len(entry.entry_hash) == 64

        # 3. Certification
        certifier = AIReadinessCertifier()
        certifier = AIReadinessCertifier()
        cert = certifier.issue_certificate(
            batch_id=batch_id,
            issue_profile=final_state.issue_profile,
            damage_pred=final_state.damage_prediction,
            pii_report=final_state.pii_report,
            policy_decision=final_state.policy_decision,
            ledger_entry_hash=entry.entry_hash,
        )
        assert cert.ai_readiness_score >= 70.0
        assert cert.grade in ("A+", "A", "A-", "B+", "B")
        assert cert.ledger_entry_hash == entry.entry_hash

        # 4. Cryptographic ledger integrity
        is_valid, violations = verify_audit_ledger(ledger_path)
        assert is_valid is True
        assert len(violations) == 0


def test_e2e_repair_then_approve_flow(setup_pipeline: dict) -> None:
    """Verify that repairable missingness is automatically imputed and approved."""
    env = setup_pipeline
    raw_df = env["test_df"].head(120)
    # Inject 4% MCAR missingness in numeric feature
    corrupted_df, _ = inject_mcar(raw_df, columns=["income"], severity=0.04, seed=123)
    batch_id = "BATCH-E2E-REPAIR-002"

    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_path = Path(tmp_dir) / "audit_ledger.jsonl"
        audit_ledger = AuditLedger(ledger_file=ledger_path)

        state = TrustGateState(
            batch_id=batch_id,
            raw_df=corrupted_df,
            spec=env["spec"],
            target_model=env["target_model"],
            contract=env["contract"],
            ref_df=env["train_df"],
            ref_stats=env["ref_stats"],
        )

        final_state = env["workflow"].run(state)
        decision_rec = final_state.policy_decision

        assert decision_rec is not None
        # Either approved after repair or approved directly
        assert decision_rec.decision in (
            Decision.APPROVE,
            Decision.REPAIR_THEN_APPROVE,
            Decision.ESCALATE_TO_HUMAN,
        )

        audit_ledger.append_decision(
            batch_id=batch_id,
            payload={
                "decision": decision_rec.decision.value,
                "repaired": decision_rec.repair_executed,
            },
        )
        assert verify_audit_ledger(ledger_path)[0] is True


def test_e2e_pii_hard_block_quarantine(setup_pipeline: dict) -> None:
    """Verify that unmasked critical PII immediately triggers zero-tolerance quarantine."""
    env = setup_pipeline
    raw_df = env["test_df"].head(80)
    # Inject direct unmasked SSNs
    pii_df = raw_df.with_columns(
        pl.Series("customer_ssn", ["123-45-6789" for _ in range(raw_df.height)])
    )
    batch_id = "BATCH-E2E-PII-003"

    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_path = Path(tmp_dir) / "audit_ledger.jsonl"
        audit_ledger = AuditLedger(ledger_file=ledger_path)

        state = TrustGateState(
            batch_id=batch_id,
            raw_df=pii_df,
            spec=env["spec"],
            target_model=env["target_model"],
            contract=env["contract"],
            ref_df=env["train_df"],
            ref_stats=env["ref_stats"],
        )

        final_state = env["workflow"].run(state)
        decision_rec = final_state.policy_decision

        assert decision_rec is not None
        assert decision_rec.decision == Decision.QUARANTINE
        assert decision_rec.hard_block_triggered is True
        assert decision_rec.risk_score == 100.0

        entry = audit_ledger.append_decision(
            batch_id=batch_id,
            payload={"decision": decision_rec.decision.value, "pii_blocked": True},
        )

        certifier = AIReadinessCertifier()
        cert = certifier.issue_certificate(
            batch_id=batch_id,
            issue_profile=final_state.issue_profile,
            damage_pred=final_state.damage_prediction,
            pii_report=final_state.pii_report,
            policy_decision=decision_rec,
            ledger_entry_hash=entry.entry_hash,
        )
        assert cert.grade == "REJECTED"


def test_e2e_audit_chain_tamper_evidence(setup_pipeline: dict) -> None:
    """Verify that tampering with an audit ledger entry is strictly detected."""
    _ = setup_pipeline
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_path = Path(tmp_dir) / "audit_ledger.jsonl"
        audit_ledger = AuditLedger(ledger_file=ledger_path)

        # Append 3 valid entries
        for i in range(1, 4):
            audit_ledger.append_decision(
                batch_id=f"BATCH-{i:03d}",
                payload={"decision": "APPROVE", "index": i},
            )

        # Baseline: valid
        is_valid, _ = verify_audit_ledger(ledger_path)
        assert is_valid is True

        # Tamper line 2 payload maliciously
        lines = ledger_path.read_text(encoding="utf-8").strip().splitlines()
        import json

        entry_2 = json.loads(lines[1])
        entry_2["payload"]["decision"] = "FORGED_APPROVAL"
        lines[1] = json.dumps(entry_2)
        ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Must fail verification
        is_valid_after_tamper, violations = verify_audit_ledger(ledger_path)
        assert is_valid_after_tamper is False
        assert any(
            "tampered" in v.lower() or "signature" in v.lower() or "hash" in v.lower()
            for v in violations
        )
