"""Unit tests for Phase 8 REST API, role authentication, background queue, and endpoints."""

import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from trustgate.api.app import (
    AuditVerifyResponse,
    HumanReviewRequest,
    JsonBatchRequest,
    SubmitBatchResponse,
    get_batch_certificate,
    get_batch_decision,
    get_batch_status,
    get_contract,
    healthz,
    list_batches,
    metrics,
    register_contract,
    review_batch,
    submit_batch_json,
    verify_audit,
)
from trustgate.api.auth import AuthenticatedUser, Role, get_current_user, require_role
from trustgate.api.queue import JobQueue, JobStatus
from trustgate.contracts.models import ColumnContract, ColumnDType, DataContract
from trustgate.simulator.datasets import get_dataset


@pytest.fixture
def test_queue(tmp_path: Path) -> JobQueue:
    """Create a clean JobQueue instance backed by a temporary test storage directory."""
    return JobQueue(storage_dir=tmp_path / "api_data")


def test_healthz_and_metrics_endpoints(test_queue: JobQueue) -> None:
    """Verify public healthz and Prometheus metrics endpoints."""
    res_health = healthz()
    assert res_health["status"] == "ok"
    assert res_health["service"] == "TrustGate"
    assert "timestamp" in res_health

    res_metrics = metrics(queue=test_queue)
    assert res_metrics.status_code == 200
    metrics_text = res_metrics.body.decode("utf-8")
    assert "trustgate_batches_total" in metrics_text
    assert "trustgate_processing_seconds" in metrics_text
    assert "trustgate_risk_score_sum" in metrics_text


def test_api_key_authentication_and_rbac() -> None:
    """Verify role-based access control and unauthorized error handling."""
    # 1. No key -> 401
    with pytest.raises(HTTPException) as exc_none:
        get_current_user(None)
    assert exc_none.value.status_code == 401
    assert "Missing" in exc_none.value.detail

    # 2. Invalid key -> 401
    with pytest.raises(HTTPException) as exc_inv:
        get_current_user("invalid-secret-key-12345")
    assert exc_inv.value.status_code == 401
    assert "Invalid API Key" in exc_inv.value.detail

    # 3. Valid users
    admin_user = get_current_user("tg-admin-secret-key")
    assert admin_user.role == Role.ADMIN

    operator_user = get_current_user("tg-operator-secret-key")
    assert operator_user.role == Role.OPERATOR

    viewer_user = get_current_user("tg-viewer-secret-key")
    assert viewer_user.role == Role.VIEWER

    # 4. RBAC enforcement
    operator_gate = require_role(Role.OPERATOR)
    admin_gate = require_role(Role.ADMIN)

    # Viewer fails operator gate
    with pytest.raises(HTTPException) as exc_rbac:
        operator_gate(viewer_user)
    assert exc_rbac.value.status_code == 403
    assert "requires minimum role 'operator'" in exc_rbac.value.detail

    # Operator passes operator gate, but fails admin gate
    assert operator_gate(operator_user) == operator_user
    with pytest.raises(HTTPException) as exc_adm:
        admin_gate(operator_user)
    assert exc_adm.value.status_code == 403

    # Admin passes both
    assert operator_gate(admin_user) == admin_user
    assert admin_gate(admin_user) == admin_user


def test_contract_lifecycle(test_queue: JobQueue) -> None:
    """Verify contract registration and retrieval."""
    contract = DataContract(
        dataset_id="credit_contract_v1",
        name="Credit Default Production Contract",
        version="1.0.0",
        owner="risk_team",
        columns=[
            ColumnContract(name="income", dtype=ColumnDType.NUMERIC, required=True, nullable=False)
        ],
    )

    # Register
    res = register_contract(contract=contract, queue=test_queue)
    assert res["status"] == "registered"
    assert res["contract_id"] == "credit_contract_v1"

    # Retrieve
    fetched = get_contract(contract_id="credit_contract_v1", queue=test_queue)
    assert fetched.dataset_id == "credit_contract_v1"
    assert fetched.name == "Credit Default Production Contract"

    # 404 for unknown
    with pytest.raises(HTTPException) as exc_404:
        get_contract(contract_id="non_existent_contract", queue=test_queue)
    assert exc_404.value.status_code == 404


def test_batch_submission_and_lifecycle(test_queue: JobQueue) -> None:
    """Verify batch submission, asynchronous background execution, and result retrieval."""
    _, test_df, _, _ = get_dataset("credit_default", n_samples=100, seed=42)
    records = test_df.head(20).to_dicts()

    payload = JsonBatchRequest(data=records, target_model="credit_default_lgb")
    submit_res: SubmitBatchResponse = submit_batch_json(payload=payload, queue=test_queue)

    assert submit_res.job_id.startswith("job_")
    assert submit_res.batch_id.startswith("batch_")
    assert submit_res.status in (JobStatus.PENDING, JobStatus.RUNNING)

    # Wait for background thread pool execution to complete
    job = test_queue.get_job(submit_res.job_id)
    assert job is not None
    for _ in range(40):
        if job.status in (
            JobStatus.COMPLETED,
            JobStatus.ESCALATED_WAITING_FOR_REVIEW,
            JobStatus.FAILED,
        ):
            break
        time.sleep(0.1)

    assert job.status in (
        JobStatus.COMPLETED,
        JobStatus.ESCALATED_WAITING_FOR_REVIEW,
    ), f"Job failed with error: {job.error}"
    assert job.decision in ("APPROVE", "REPAIR_THEN_APPROVE", "QUARANTINE", "ESCALATE_TO_HUMAN")
    assert job.risk_score is not None
    assert job.damage_upper is not None
    assert job.ai_readiness_score is not None

    # Status endpoint
    status_resp = get_batch_status(job_id=submit_res.job_id, queue=test_queue)
    assert status_resp.decision == job.decision
    assert status_resp.risk_score == job.risk_score

    # Decision endpoint
    dec_resp = get_batch_decision(job_id=submit_res.job_id, queue=test_queue)
    assert dec_resp.job_id == submit_res.job_id
    assert dec_resp.decision == job.decision
    assert isinstance(dec_resp.top_drivers, list)
    assert dec_resp.explanation is not None

    # Certificate endpoint
    cert_data = get_batch_certificate(job_id=submit_res.job_id, queue=test_queue)
    assert "ai_readiness_score" in cert_data
    assert "grade" in cert_data
    assert "tamper_evident_seal" in cert_data

    # List batches endpoint
    batch_list = list_batches(queue=test_queue)
    assert len(batch_list) >= 1
    assert any(b.job_id == submit_res.job_id for b in batch_list)


def test_human_review_workflow(test_queue: JobQueue) -> None:
    """Verify human sign-off action and ledger recording."""
    _, test_df, _, _ = get_dataset("credit_default", n_samples=50, seed=42)
    records = test_df.head(10).to_dicts()

    payload = JsonBatchRequest(data=records)
    sub = submit_batch_json(payload=payload, queue=test_queue)

    admin_user = AuthenticatedUser(
        api_key="tg-admin-secret-key", role=Role.ADMIN, description="Admin"
    )

    # Reviewer approves
    rev_req = HumanReviewRequest(approved=True, comment="Business override approved.")
    res = review_batch(
        job_id=sub.job_id,
        review=rev_req,
        current_user=admin_user,
        queue=test_queue,
    )
    assert res.status == JobStatus.COMPLETED
    assert res.decision == "APPROVE"

    job = test_queue.get_job(sub.job_id)
    assert job is not None
    assert job.human_review is not None
    assert job.human_review["approved"] is True


def test_audit_ledger_verification_endpoint(test_queue: JobQueue) -> None:
    """Verify GET /v1/audit/verify endpoint on audit ledger."""
    # Add a mock entry
    test_queue.audit_ledger.append_decision(
        batch_id="batch_verify_test",
        payload={"action": "INIT_TEST", "val": 100},
    )

    verify_res: AuditVerifyResponse = verify_audit(queue=test_queue)
    assert verify_res.valid is True
    assert verify_res.entry_count >= 1
    assert len(verify_res.violations) == 0


def test_empty_batch_submission_rejection(test_queue: JobQueue) -> None:
    """Verify that empty batch data arrays are rejected with 400 Bad Request."""
    empty_payload = JsonBatchRequest(data=[])
    with pytest.raises(HTTPException) as exc_empty:
        submit_batch_json(payload=empty_payload, queue=test_queue)
    assert exc_empty.value.status_code == 400
    assert "cannot be empty" in exc_empty.value.detail
