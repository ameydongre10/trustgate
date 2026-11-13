"""In-process asynchronous job queue and execution engine for TrustGate."""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from trustgate.agents.graph import TrustGateWorkflow
from trustgate.agents.llm_clients import TemplateLLMClient
from trustgate.agents.state import TrustGateState
from trustgate.audit.ledger import AuditLedger
from trustgate.audit.lineage import LineageGraph
from trustgate.certify.ai_readiness import AIReadinessCertifier
from trustgate.contracts.models import DataContract
from trustgate.damage.features import DAMAGE_FEATURE_NAMES
from trustgate.damage.predictor import DamagePredictor
from trustgate.ingest.readers import read_batch_file
from trustgate.policy.engine import Decision, PolicyEngine
from trustgate.simulator.datasets import get_dataset
from trustgate.simulator.models import TargetModel, train_target_model


class JobStatus(StrEnum):
    """Lifecycle status of a batch governance job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ESCALATED_WAITING_FOR_REVIEW = "ESCALATED_WAITING_FOR_REVIEW"


class JobRecord(BaseModel):
    """Complete persistent state of an enqueued or executed batch job."""

    job_id: str
    batch_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    contract_id: str | None = None
    target_model_name: str = "credit_default_lgb"
    error: str | None = None
    decision: str | None = None
    risk_score: float | None = None
    predicted_damage: float | None = None
    damage_upper: float | None = None
    ai_readiness_score: float | None = None
    ai_readiness_grade: str | None = None
    certificate_id: str | None = None
    certificate: dict[str, Any] | None = None
    explanation: str | None = None
    human_review: dict[str, Any] | None = None
    profile_summary: dict[str, Any] | None = None
    top_drivers: list[dict[str, Any]] = Field(default_factory=list)
    repaired_cells_count: int = 0
    dirty_preview: list[dict[str, Any]] = Field(default_factory=list)
    repaired_preview: list[dict[str, Any]] = Field(default_factory=list)


class JobQueue:
    """Thread-safe in-memory job repository and execution worker."""

    def __init__(
        self,
        storage_dir: Path | str | None = None,
        max_workers: int = 2,
        damage_predictor: DamagePredictor | None = None,
        target_models: dict[str, TargetModel] | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir or "trustgate_data")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_file = self.storage_dir / "audit_ledger.jsonl"
        self.audit_ledger = AuditLedger(ledger_file=self.ledger_file)
        self.certifier = AIReadinessCertifier()

        self.jobs: dict[str, JobRecord] = {}
        self.contracts: dict[str, DataContract] = {}
        self.target_models: dict[str, TargetModel] = dict(target_models) if target_models else {}
        self.damage_predictor: DamagePredictor | None = damage_predictor
        self.policy_engine = PolicyEngine()
        self.llm_client = TemplateLLMClient()

        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Prometheus metrics tracking counters
        self.metrics_batches_total: dict[str, int] = {
            Decision.APPROVE.value: 0,
            Decision.REPAIR_THEN_APPROVE.value: 0,
            Decision.QUARANTINE.value: 0,
            Decision.ESCALATE_TO_HUMAN.value: 0,
            "FAILED": 0,
        }
        self.metrics_processing_seconds_sum: float = 0.0
        self.metrics_processing_count: int = 0
        self.metrics_risk_score_sum: float = 0.0
        self.metrics_damage_avoided_sum: float = 0.0

        if not self.target_models or self.damage_predictor is None:
            self._initialize_builtins()

    def _initialize_builtins(self) -> None:
        """Initialize lightweight default TargetModel and DamagePredictor for the service."""
        try:
            if not self.target_models:
                train_df, _, spec, ref_stats = get_dataset("credit_default", n_samples=150, seed=42)
                model = train_target_model(
                    spec, train_df, ref_stats, model_type="lightgbm", seed=42
                )
                self.target_models["credit_default_lgb"] = model

            if self.damage_predictor is None:
                rng = np.random.RandomState(42)
                X = rng.uniform(0.0, 0.3, size=(60, len(DAMAGE_FEATURE_NAMES)))
                # Realistic damage mapping with noise
                y = np.clip(
                    X[:, 2] * 0.4 + X[:, 5] * 0.3 + X[:, 0] * 0.1 + rng.normal(0, 0.01, size=60),
                    0.0,
                    1.0,
                )
                self.damage_predictor = DamagePredictor(n_estimators=25, seed=42).fit(X, y)
        except Exception:
            pass

    def register_contract(self, contract: DataContract) -> None:
        """Register or update a data contract."""
        self.contracts[contract.dataset_id] = contract
        self.contracts[contract.name] = contract

    def get_contract(self, contract_id: str) -> DataContract | None:
        """Lookup registered data contract."""
        return self.contracts.get(contract_id)

    def submit_batch(
        self,
        raw_bytes: bytes,
        filename: str,
        contract_id: str | None = None,
        target_model_name: str = "credit_default_lgb",
    ) -> JobRecord:
        """Enqueue incoming data batch for processing."""
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"

        record = JobRecord(
            job_id=job_id,
            batch_id=batch_id,
            status=JobStatus.PENDING,
            contract_id=contract_id,
            target_model_name=target_model_name,
        )
        self.jobs[job_id] = record

        # Dispatch async execution in background thread pool
        self.executor.submit(
            self._process_job,
            job_id=job_id,
            raw_bytes=raw_bytes,
            filename=filename,
            contract_id=contract_id,
            target_model_name=target_model_name,
        )
        return record

    def _process_job(
        self,
        job_id: str,
        raw_bytes: bytes,
        filename: str,
        contract_id: str | None,
        target_model_name: str,
    ) -> None:
        """Execute end-to-end governance workflow in background."""
        record = self.jobs[job_id]
        record.status = JobStatus.RUNNING
        record.updated_at = time.time()
        start_time = time.time()

        try:
            # 1. Parse batch data
            temp_file = self.storage_dir / f"{record.batch_id}_{filename}"
            temp_file.write_bytes(raw_bytes)

            batch_data = read_batch_file(temp_file)
            df = batch_data.df

            # Previews for dashboard
            dirty_preview = df.head(10).to_dicts()
            record.dirty_preview = dirty_preview

            # 2. Resolve Target Model and Contract
            model = self.target_models.get(target_model_name)
            if model is None:
                if "credit_default_lgb" in self.target_models:
                    model = self.target_models["credit_default_lgb"]
                else:
                    self._initialize_builtins()
                    model = self.target_models.get("credit_default_lgb")

            if model is None:
                raise RuntimeError(f"Target model '{target_model_name}' not available.")

            contract = self.contracts.get(contract_id) if contract_id else None

            # Reference data & stats from model plugin
            ref_stats = model.reference_stats
            train_df, _, _, _ = get_dataset(model.spec.name, n_samples=300, seed=42)

            # 3. Setup Lineage
            lineage = LineageGraph(batch_id=record.batch_id)
            raw_node_id = f"raw_{record.batch_id}"
            lineage.add_node(
                raw_node_id,
                "RawBatchIngest",
                {"rows": df.height, "cols": df.width, "filename": filename},
            )

            # 4. Construct initial agent state
            state = TrustGateState(
                batch_id=record.batch_id,
                raw_df=df,
                spec=model.spec,
                target_model=model,
                contract=contract,
                ref_df=train_df,
                ref_stats=ref_stats,
            )

            # 5. Execute state machine DAG
            workflow = TrustGateWorkflow(
                policy_engine=self.policy_engine,
                damage_predictor=self.damage_predictor,
                llm_client=self.llm_client,
            )
            final_state = workflow.run(state)

            duration = time.time() - start_time
            self.metrics_processing_seconds_sum += duration
            self.metrics_processing_count += 1

            # 6. Extract results & update job
            decision_rec = final_state.policy_decision
            if decision_rec is None:
                raise RuntimeError("Workflow finished without policy decision.")

            record.decision = decision_rec.decision.value
            record.risk_score = decision_rec.risk_score
            record.predicted_damage = decision_rec.predicted_damage
            record.damage_upper = decision_rec.damage_upper
            record.explanation = final_state.explanation or ""

            # Profile summary
            if final_state.issue_profile:
                record.profile_summary = {
                    "row_count": final_state.issue_profile.get("row_count", df.height),
                    "column_count": final_state.issue_profile.get("column_count", df.width),
                    "max_null_rate": final_state.issue_profile.get("max_null_rate", 0.0),
                    "max_psi": final_state.issue_profile.get("max_psi", 0.0),
                    "outlier_rate": final_state.issue_profile.get("outlier_rate", 0.0),
                    "pii_findings_count": (
                        final_state.pii_report.total_findings_count if final_state.pii_report else 0
                    ),
                }

            # Top SHAP drivers
            if final_state.damage_prediction and final_state.damage_prediction.top_factors:
                record.top_drivers = [
                    {"driver": d[0], "impact": float(d[1])}
                    for d in final_state.damage_prediction.top_factors
                ]

            # Repaired data preview
            if final_state.repaired_df is not None:
                record.repaired_preview = final_state.repaired_df.head(10).to_dicts()
                if final_state.repair_plan:
                    record.repaired_cells_count = final_state.repair_plan.total_cells_modified

            # 7. Append decision to tamper-evident audit ledger
            audit_entry = self.audit_ledger.append_decision(
                batch_id=record.batch_id,
                payload={
                    "job_id": record.job_id,
                    "decision": record.decision,
                    "risk_score": record.risk_score,
                    "damage_upper": record.damage_upper,
                },
            )

            # 8. Generate AI Readiness Certificate sealed with ledger entry hash
            cert = self.certifier.issue_certificate(
                batch_id=record.batch_id,
                issue_profile=final_state.issue_profile,
                damage_pred=final_state.damage_prediction,
                pii_report=final_state.pii_report,
                policy_decision=decision_rec,
                ledger_entry_hash=audit_entry.entry_hash,
            )
            record.ai_readiness_score = cert.ai_readiness_score
            record.ai_readiness_grade = cert.grade
            record.certificate_id = cert.certificate_id
            record.certificate = cert.model_dump()

            # Update metrics
            if record.decision:
                self.metrics_batches_total[record.decision] = (
                    self.metrics_batches_total.get(record.decision, 0) + 1
                )
            self.metrics_risk_score_sum += record.risk_score or 0.0
            if record.decision in (Decision.QUARANTINE.value, Decision.REPAIR_THEN_APPROVE.value):
                self.metrics_damage_avoided_sum += record.damage_upper or 0.0

            # 9. Handle human review status
            if decision_rec.decision == Decision.ESCALATE_TO_HUMAN:
                record.status = JobStatus.ESCALATED_WAITING_FOR_REVIEW
            else:
                record.status = JobStatus.COMPLETED

        except Exception as e:
            record.status = JobStatus.FAILED
            record.error = str(e)
            record.decision = Decision.QUARANTINE.value
            self.metrics_batches_total["FAILED"] += 1
        finally:
            record.updated_at = time.time()

    def review_job(
        self,
        job_id: str,
        approved: bool,
        comment: str,
        reviewer: str = "operator",
    ) -> JobRecord:
        """Apply human sign-off action on an escalated batch."""
        record = self.jobs.get(job_id)
        if not record:
            raise ValueError(f"Job {job_id} not found.")

        # Update record with human review action
        now = datetime.now(UTC).timestamp()
        record.human_review = {
            "approved": approved,
            "comment": comment,
            "reviewer": reviewer,
            "timestamp": now,
        }

        # Override decision based on sign-off
        if approved:
            record.decision = Decision.APPROVE.value
            record.status = JobStatus.COMPLETED
            if record.explanation:
                record.explanation += f"\n\n**Operator Override:** Batch manually approved by {reviewer}. Comment: {comment}"
        else:
            record.decision = Decision.QUARANTINE.value
            record.status = JobStatus.COMPLETED
            if record.explanation:
                record.explanation += (
                    f"\n\n**Operator Decision:** Batch quarantined by {reviewer}. Reason: {comment}"
                )

        # Append human review action to audit ledger
        self.audit_ledger.append_decision(
            batch_id=record.batch_id,
            payload={
                "action": "HUMAN_REVIEW_OVERRIDE",
                "approved": approved,
                "reviewer": reviewer,
                "comment": comment,
                "final_decision": record.decision,
            },
        )
        record.updated_at = time.time()
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        """Lookup job record."""
        return self.jobs.get(job_id)

    def list_jobs(self, status: JobStatus | None = None) -> list[JobRecord]:
        """List all jobs optionally filtered by status."""
        if status is None:
            return list(self.jobs.values())
        return [j for j in self.jobs.values() if j.status == status]

    def get_prometheus_metrics(self) -> str:
        """Generate plain-text Prometheus formatted metrics."""
        lines = [
            "# HELP trustgate_batches_total Total count of evaluated data batches partitioned by governance decision.",
            "# TYPE trustgate_batches_total counter",
        ]
        for dec, count in self.metrics_batches_total.items():
            lines.append(f'trustgate_batches_total{{decision="{dec}"}} {count}')

        lines.extend(
            [
                "# HELP trustgate_processing_seconds Duration of batch governance processing in seconds.",
                "# TYPE trustgate_processing_seconds summary",
                f"trustgate_processing_seconds_sum {self.metrics_processing_seconds_sum:.4f}",
                f"trustgate_processing_seconds_count {self.metrics_processing_count}",
                "# HELP trustgate_risk_score_sum Sum of computed risk scores across all batches.",
                "# TYPE trustgate_risk_score_sum counter",
                f"trustgate_risk_score_sum {self.metrics_risk_score_sum:.2f}",
                "# HELP trustgate_damage_avoided_sum Estimated downstream damage percentage avoided via quarantine or repair.",
                "# TYPE trustgate_damage_avoided_sum counter",
                f"trustgate_damage_avoided_sum {self.metrics_damage_avoided_sum:.4f}",
            ]
        )
        return "\n".join(lines) + "\n"
