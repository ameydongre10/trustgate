"""FastAPI application for TrustGate Data Governance Engine."""

import io
import os
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from trustgate.api.auth import AuthenticatedUser, Role, require_role
from trustgate.api.queue import JobQueue, JobStatus
from trustgate.audit.ledger import verify_audit_ledger
from trustgate.contracts.models import DataContract

app = FastAPI(
    title="TrustGate API",
    version="0.1.0",
    description="Impact-Aware Data Trust and Verification Engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global JobQueue singleton instance
queue_instance: JobQueue | None = None


def get_queue() -> JobQueue:
    """Dependency provider for the job queue and execution engine."""
    global queue_instance
    if queue_instance is None:
        storage_dir = os.getenv("TRUSTGATE_STORAGE_DIR", "trustgate_data")
        queue_instance = JobQueue(storage_dir=storage_dir)
    return queue_instance


# ---------------------------------------------------------------------------
# Request and Response Schemas
# ---------------------------------------------------------------------------


class SubmitBatchResponse(BaseModel):
    job_id: str
    batch_id: str
    status: JobStatus
    created_at: float
    target_model: str


class JsonBatchRequest(BaseModel):
    data: list[dict[str, Any]]
    target_model: str = "credit_default_lgb"
    contract_id: str | None = None


class BatchStatusResponse(BaseModel):
    job_id: str
    batch_id: str
    status: JobStatus
    created_at: float
    updated_at: float
    decision: str | None = None
    risk_score: float | None = None
    predicted_damage: float | None = None
    damage_upper: float | None = None
    ai_readiness_score: float | None = None
    ai_readiness_grade: str | None = None
    profile_summary: dict[str, Any] | None = None
    certificate_url: str | None = None
    error: str | None = None


class BatchDecisionResponse(BaseModel):
    batch_id: str
    job_id: str
    decision: str | None
    risk_score: float | None
    predicted_damage: float | None
    damage_upper: float | None
    top_drivers: list[dict[str, Any]]
    explanation: str | None
    repaired_cells_count: int
    human_review: dict[str, Any] | None


class HumanReviewRequest(BaseModel):
    approved: bool
    comment: str = Field(..., min_length=3, max_length=1000)


class AuditVerifyResponse(BaseModel):
    valid: bool
    entry_count: int
    head_hash: str
    violations: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz", tags=["Observability"])
def healthz() -> dict[str, Any]:
    """Public health check endpoint."""
    return {
        "status": "ok",
        "service": "TrustGate",
        "version": "0.1.0",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/metrics", tags=["Observability"])
def metrics(queue: Annotated[JobQueue, Depends(get_queue)]) -> Response:
    """Public Prometheus formatted metrics endpoint."""
    metrics_text = queue.get_prometheus_metrics()
    return Response(content=metrics_text, media_type="text/plain; version=0.0.4")


@app.post(
    "/v1/contracts",
    response_model=dict[str, Any],
    tags=["Contracts"],
    dependencies=[Depends(require_role(Role.OPERATOR))],
)
def register_contract(
    contract: DataContract,
    queue: Annotated[JobQueue, Depends(get_queue)],
) -> dict[str, Any]:
    """Register or update a data contract."""
    queue.register_contract(contract)
    return {
        "status": "registered",
        "contract_id": contract.dataset_id,
        "name": contract.name,
        "version": contract.version,
    }


@app.get(
    "/v1/contracts/{contract_id}",
    response_model=DataContract,
    tags=["Contracts"],
    dependencies=[Depends(require_role(Role.VIEWER))],
)
def get_contract(
    contract_id: str,
    queue: Annotated[JobQueue, Depends(get_queue)],
) -> DataContract:
    """Retrieve registered contract by ID."""
    contract = queue.get_contract(contract_id)
    if not contract:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contract '{contract_id}' not found.",
        )
    return contract


@app.post(
    "/v1/batches",
    response_model=SubmitBatchResponse,
    tags=["Batches"],
    dependencies=[Depends(require_role(Role.OPERATOR))],
)
async def submit_batch_file(
    queue: Annotated[JobQueue, Depends(get_queue)],
    file: UploadFile = File(...),
    target_model: str = Form("credit_default_lgb"),
    contract_id: str | None = Form(None),
) -> SubmitBatchResponse:
    """Submit a tabular batch file (CSV or Parquet) for governance evaluation."""
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    filename = file.filename or "batch.csv"
    job = queue.submit_batch(
        raw_bytes=content,
        filename=filename,
        contract_id=contract_id,
        target_model_name=target_model,
    )

    return SubmitBatchResponse(
        job_id=job.job_id,
        batch_id=job.batch_id,
        status=job.status,
        created_at=job.created_at,
        target_model=job.target_model_name,
    )


@app.post(
    "/v1/batches/json",
    response_model=SubmitBatchResponse,
    tags=["Batches"],
    dependencies=[Depends(require_role(Role.OPERATOR))],
)
def submit_batch_json(
    payload: JsonBatchRequest,
    queue: Annotated[JobQueue, Depends(get_queue)],
) -> SubmitBatchResponse:
    """Submit a list of JSON records directly for governance evaluation."""
    if not payload.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload data array cannot be empty.",
        )

    import polars as pl

    df = pl.from_dicts(payload.data)
    csv_buf = io.BytesIO()
    df.write_csv(csv_buf)
    raw_bytes = csv_buf.getvalue()

    job = queue.submit_batch(
        raw_bytes=raw_bytes,
        filename="batch_records.csv",
        contract_id=payload.contract_id,
        target_model_name=payload.target_model,
    )

    return SubmitBatchResponse(
        job_id=job.job_id,
        batch_id=job.batch_id,
        status=job.status,
        created_at=job.created_at,
        target_model=job.target_model_name,
    )


@app.get(
    "/v1/batches",
    response_model=list[BatchStatusResponse],
    tags=["Batches"],
    dependencies=[Depends(require_role(Role.VIEWER))],
)
def list_batches(
    queue: Annotated[JobQueue, Depends(get_queue)],
    status_filter: JobStatus | None = None,
) -> list[BatchStatusResponse]:
    """List submitted batch jobs with optional status filtering."""
    jobs = queue.list_jobs(status=status_filter)
    results = []
    for j in jobs:
        cert_url = f"/v1/batches/{j.job_id}/certificate" if j.certificate_id else None
        results.append(
            BatchStatusResponse(
                job_id=j.job_id,
                batch_id=j.batch_id,
                status=j.status,
                created_at=j.created_at,
                updated_at=j.updated_at,
                decision=j.decision,
                risk_score=j.risk_score,
                predicted_damage=j.predicted_damage,
                damage_upper=j.damage_upper,
                ai_readiness_score=j.ai_readiness_score,
                ai_readiness_grade=j.ai_readiness_grade,
                profile_summary=j.profile_summary,
                certificate_url=cert_url,
                error=j.error,
            )
        )
    return results


@app.get(
    "/v1/batches/{job_id}",
    response_model=BatchStatusResponse,
    tags=["Batches"],
    dependencies=[Depends(require_role(Role.VIEWER))],
)
def get_batch_status(
    job_id: str,
    queue: Annotated[JobQueue, Depends(get_queue)],
) -> BatchStatusResponse:
    """Retrieve execution status and summary profile for a batch job."""
    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job '{job_id}' not found.",
        )

    cert_url = f"/v1/batches/{job.job_id}/certificate" if job.certificate_id else None
    return BatchStatusResponse(
        job_id=job.job_id,
        batch_id=job.batch_id,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        decision=job.decision,
        risk_score=job.risk_score,
        predicted_damage=job.predicted_damage,
        damage_upper=job.damage_upper,
        ai_readiness_score=job.ai_readiness_score,
        ai_readiness_grade=job.ai_readiness_grade,
        profile_summary=job.profile_summary,
        certificate_url=cert_url,
        error=job.error,
    )


@app.get(
    "/v1/batches/{job_id}/decision",
    response_model=BatchDecisionResponse,
    tags=["Batches"],
    dependencies=[Depends(require_role(Role.VIEWER))],
)
def get_batch_decision(
    job_id: str,
    queue: Annotated[JobQueue, Depends(get_queue)],
) -> BatchDecisionResponse:
    """Retrieve full governance decision record, predicted damage bounds, and SHAP explanation."""
    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job '{job_id}' not found.",
        )

    return BatchDecisionResponse(
        batch_id=job.batch_id,
        job_id=job.job_id,
        decision=job.decision,
        risk_score=job.risk_score,
        predicted_damage=job.predicted_damage,
        damage_upper=job.damage_upper,
        top_drivers=job.top_drivers,
        explanation=job.explanation,
        repaired_cells_count=job.repaired_cells_count,
        human_review=job.human_review,
    )


@app.post(
    "/v1/batches/{job_id}/review",
    response_model=BatchStatusResponse,
    tags=["Batches"],
    dependencies=[Depends(require_role(Role.OPERATOR))],
)
def review_batch(
    job_id: str,
    review: HumanReviewRequest,
    current_user: Annotated[AuthenticatedUser, Depends(require_role(Role.OPERATOR))],
    queue: Annotated[JobQueue, Depends(get_queue)],
) -> BatchStatusResponse:
    """Apply human sign-off action (approve override or quarantine confirm) on an escalated batch."""
    try:
        updated_job = queue.review_job(
            job_id=job_id,
            approved=review.approved,
            comment=review.comment,
            reviewer=f"{current_user.role.value}:{current_user.description}",
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    cert_url = (
        f"/v1/batches/{updated_job.job_id}/certificate" if updated_job.certificate_id else None
    )
    return BatchStatusResponse(
        job_id=updated_job.job_id,
        batch_id=updated_job.batch_id,
        status=updated_job.status,
        created_at=updated_job.created_at,
        updated_at=updated_job.updated_at,
        decision=updated_job.decision,
        risk_score=updated_job.risk_score,
        predicted_damage=updated_job.predicted_damage,
        damage_upper=updated_job.damage_upper,
        ai_readiness_score=updated_job.ai_readiness_score,
        ai_readiness_grade=updated_job.ai_readiness_grade,
        profile_summary=updated_job.profile_summary,
        certificate_url=cert_url,
        error=updated_job.error,
    )


@app.get(
    "/v1/batches/{job_id}/certificate",
    response_model=dict[str, Any],
    tags=["Certify"],
    dependencies=[Depends(require_role(Role.VIEWER))],
)
def get_batch_certificate(
    job_id: str,
    queue: Annotated[JobQueue, Depends(get_queue)],
) -> dict[str, Any]:
    """Retrieve issued AI-Readiness TrustCertificate."""
    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job '{job_id}' not found.",
        )
    if not job.certificate:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch job '{job_id}' has not produced a certificate yet (status: {job.status.value}).",
        )
    return job.certificate


@app.get(
    "/v1/audit/verify",
    response_model=AuditVerifyResponse,
    tags=["Audit"],
    dependencies=[Depends(require_role(Role.VIEWER))],
)
def verify_audit(
    queue: Annotated[JobQueue, Depends(get_queue)],
) -> AuditVerifyResponse:
    """Verify hash chain integrity across all historical ledger entries."""
    valid, violations = verify_audit_ledger(queue.ledger_file)
    entry_count = len(queue.audit_ledger.entries)
    head_hash = queue.audit_ledger.head_hash

    return AuditVerifyResponse(
        valid=valid,
        entry_count=entry_count,
        head_hash=head_hash,
        violations=violations,
    )
