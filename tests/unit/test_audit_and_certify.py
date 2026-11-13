"""Unit tests for Phase 7 audit ledger, hash-chain verification, lineage, and AI-readiness certification."""

import json
from pathlib import Path

from trustgate.audit.ledger import (
    GENESIS_PREV_HASH,
    AuditLedger,
    verify_audit_ledger,
)
from trustgate.audit.lineage import LineageGraph
from trustgate.certify.ai_readiness import AIReadinessCertifier, TrustCertificate
from trustgate.policy.engine import Decision, PolicyDecisionRecord


def test_audit_ledger_append_and_verify(tmp_path: Path) -> None:
    """Verify hash chaining and verification across multiple audit entries."""
    log_file = tmp_path / "test_ledger.jsonl"
    ledger = AuditLedger(ledger_file=log_file)

    # Entry 0 (Genesis)
    e0 = ledger.append_decision(
        batch_id="batch_00",
        payload={"decision": "APPROVE", "damage_upper": 0.02, "risk_score": 10.0},
    )
    assert e0.sequence_num == 0
    assert e0.prev_hash == GENESIS_PREV_HASH
    assert len(e0.entry_hash) == 64

    # Entry 1
    e1 = ledger.append_decision(
        batch_id="batch_01",
        payload={"decision": "REPAIR_THEN_APPROVE", "damage_upper": 0.04, "risk_score": 25.0},
    )
    assert e1.sequence_num == 1
    assert e1.prev_hash == e0.entry_hash

    # Entry 2
    e2 = ledger.append_decision(
        batch_id="batch_02",
        payload={"decision": "QUARANTINE", "damage_upper": 0.35, "risk_score": 95.0},
    )
    assert e2.sequence_num == 2
    assert e2.prev_hash == e1.entry_hash

    # Verification on untampered log
    is_valid, violations = verify_audit_ledger(log_file)
    assert is_valid
    assert len(violations) == 0


def test_audit_hash_chain_tamper_detection(tmp_path: Path) -> None:
    """Verify that any modification to payload, hash, or order is instantly detected."""
    log_file = tmp_path / "tamper_ledger.jsonl"
    ledger = AuditLedger(ledger_file=log_file)

    for i in range(3):
        ledger.append_decision(f"batch_{i:02d}", {"val": i, "status": "OK"})

    # Tamper with Line 2 (modify payload value)
    lines = log_file.read_text(encoding="utf-8").splitlines()
    entry1 = json.loads(lines[1])
    entry1["payload"]["val"] = 999  # Maliciously altered payload
    lines[1] = json.dumps(entry1)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Verification must fail!
    is_valid, violations = verify_audit_ledger(log_file)
    assert not is_valid
    assert any("tampered" in v.lower() for v in violations)


def test_lineage_graph_and_hashing() -> None:
    """Verify lineage node and edge tracking and graphviz rendering."""
    lineage = LineageGraph(batch_id="BATCH_100")
    lineage.add_node("raw_100", "Ingest", {"rows": 500, "cols": 10})
    lineage.add_node("profile_100", "Profile", {"null_rate": 0.05})
    lineage.add_node("decision_100", "Judge", {"decision": "APPROVE"})

    lineage.add_edge("raw_100", "profile_100", "profiled_to")
    lineage.add_edge("profile_100", "decision_100", "evaluated_by")

    graph_dict = lineage.to_dict()
    assert len(graph_dict["nodes"]) == 3
    assert len(graph_dict["edges"]) == 2

    dot_str = lineage.to_graphviz()
    assert 'digraph "Lineage_BATCH_100"' in dot_str
    assert '"raw_100" -> "profile_100"' in dot_str


def test_ai_readiness_certification() -> None:
    """Verify 10-dimensional AI-readiness scoring and certificate issuing."""
    certifier = AIReadinessCertifier()

    issue_profile = {
        "max_null_rate": 0.01,
        "type_violation_rate": 0.0,
        "max_psi": 0.02,
        "outlier_rate": 0.01,
        "duplicate_rate": 0.0,
        "schema_columns_missing": 0.0,
        "class_balance_ratio": 0.5,
    }

    decision_record = PolicyDecisionRecord(
        batch_id="batch_cert_01",
        decision=Decision.APPROVE,
        risk_score=15.0,
        predicted_damage=0.01,
        damage_upper=0.02,
        conformal_margin=0.01,
        applied_tier="standard",
        is_ood=False,
        requires_human_review=False,
        hard_block_triggered=False,
        rationale="Approved cleanly",
        timestamp=1000.0,
    )

    cert: TrustCertificate = certifier.issue_certificate(
        batch_id="batch_cert_01",
        issue_profile=issue_profile,
        damage_pred=None,
        pii_report=None,
        policy_decision=decision_record,
        ledger_entry_hash="a" * 64,
        timestamp=1000.0,
    )

    assert cert.ai_readiness_score >= 85.0
    assert cert.grade in ("A+", "A")
    assert cert.is_certified is True
    assert cert.decision == Decision.APPROVE
    assert len(cert.dimension_scores) == 10
    assert len(cert.tamper_evident_seal) == 64
