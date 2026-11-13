"""AI-Readiness scoring across 10 transparent dimensions and Trust Certificate generation."""

import hashlib
import time
from typing import Any

from pydantic import BaseModel, Field

from trustgate.damage.predictor import DamagePrediction
from trustgate.pii.scanner import PIIScanReport
from trustgate.policy.engine import Decision, PolicyDecisionRecord


class DimensionScore(BaseModel):
    name: str
    score: float = Field(..., ge=0.0, le=100.0)
    weight: float = Field(..., ge=0.0, le=1.0)
    description: str


class TrustCertificate(BaseModel):
    certificate_id: str
    batch_id: str
    ai_readiness_score: float = Field(..., ge=0.0, le=100.0)
    grade: str
    is_certified: bool
    decision: Decision
    dimension_scores: dict[str, float]
    ledger_entry_hash: str
    issued_at: float
    tamper_evident_seal: str


class AIReadinessCertifier:
    """Evaluates data trustworthiness across 10 quantitative dimensions and issues certificates."""

    WEIGHTS = {
        "completeness": 0.12,
        "validity": 0.10,
        "drift_stability": 0.15,
        "privacy_safety": 0.15,
        "outlier_resistance": 0.08,
        "uniqueness": 0.08,
        "schema_integrity": 0.10,
        "consistency": 0.05,
        "recency": 0.05,
        "model_safety": 0.12,
    }

    def compute_scores(
        self,
        issue_profile: dict[str, Any],
        damage_pred: DamagePrediction | None,
        pii_report: PIIScanReport | None,
        policy_decision: PolicyDecisionRecord,
    ) -> tuple[float, str, dict[str, float]]:
        """Compute transparent dimensional scores and overall AI-readiness score in [0, 100]."""
        # If batch was quarantined, score is capped / rejected
        if policy_decision.decision == Decision.QUARANTINE:
            grade = "REJECTED"
        dim_scores: dict[str, float] = {}

        # 1. Completeness
        max_null = float(issue_profile.get("max_null_rate", 0.0))
        dim_scores["completeness"] = max(0.0, min(100.0, round((1.0 - max_null) * 100.0, 1)))

        # 2. Validity
        type_err = float(issue_profile.get("type_violation_rate", 0.0))
        dim_scores["validity"] = max(0.0, min(100.0, round((1.0 - type_err) * 100.0, 1)))

        # 3. Drift Stability
        max_psi = float(issue_profile.get("max_psi", 0.0))
        # PSI > 0.25 is severe drift
        dim_scores["drift_stability"] = max(
            0.0, min(100.0, round((1.0 - min(1.0, max_psi / 0.25)) * 100.0, 1))
        )

        # 4. Privacy Safety
        if pii_report and (pii_report.has_critical_pii or pii_report.contains_unmasked_direct_pii):
            dim_scores["privacy_safety"] = 0.0
        elif pii_report and pii_report.has_high_pii:
            dim_scores["privacy_safety"] = 40.0
        elif pii_report and pii_report.total_findings_count > 0:
            dim_scores["privacy_safety"] = 75.0
        else:
            dim_scores["privacy_safety"] = 100.0

        # 5. Outlier Resistance
        outliers = float(issue_profile.get("outlier_rate", 0.0))
        dim_scores["outlier_resistance"] = max(
            0.0, min(100.0, round((1.0 - min(1.0, outliers * 5.0)) * 100.0, 1))
        )

        # 6. Uniqueness
        dupe_rate = float(issue_profile.get("duplicate_rate", 0.0))
        dim_scores["uniqueness"] = max(0.0, min(100.0, round((1.0 - dupe_rate) * 100.0, 1)))

        # 7. Schema Integrity
        missing_cols = float(issue_profile.get("schema_columns_missing", 0.0))
        dim_scores["schema_integrity"] = max(0.0, min(100.0, round(100.0 - missing_cols * 30.0, 1)))

        # 8. Consistency
        cb_ratio = float(issue_profile.get("class_balance_ratio", 0.5))
        dim_scores["consistency"] = 100.0 if (0.05 <= cb_ratio <= 0.95) else 40.0

        # 9. Recency
        dim_scores["recency"] = 100.0

        # 10. Model Safety
        if damage_pred:
            d_upper = damage_pred.damage_upper
            dim_scores["model_safety"] = max(
                0.0, min(100.0, round((1.0 - min(1.0, d_upper / 0.20)) * 100.0, 1))
            )
        else:
            dim_scores["model_safety"] = 50.0

        # Compute weighted average
        total_score = sum(dim_scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        final_score = round(total_score, 1)

        # Assign Grade
        if policy_decision.decision == Decision.QUARANTINE:
            grade = "REJECTED"
        elif final_score >= 90.0:
            grade = "A+"
        elif final_score >= 80.0:
            grade = "A"
        elif final_score >= 70.0:
            grade = "B"
        elif final_score >= 60.0:
            grade = "C"
        else:
            grade = "REJECTED"

        return final_score, grade, dim_scores

    def issue_certificate(
        self,
        batch_id: str,
        issue_profile: dict[str, Any],
        damage_pred: DamagePrediction | None,
        pii_report: PIIScanReport | None,
        policy_decision: PolicyDecisionRecord,
        ledger_entry_hash: str,
        timestamp: float | None = None,
    ) -> TrustCertificate:
        """Issue a tamper-evident cryptographic Trust Certificate."""
        score, grade, dim_scores = self.compute_scores(
            issue_profile=issue_profile,
            damage_pred=damage_pred,
            pii_report=pii_report,
            policy_decision=policy_decision,
        )

        ts = timestamp if timestamp is not None else time.time()
        is_certified = (grade in ("A+", "A", "B")) and (
            policy_decision.decision != Decision.QUARANTINE
        )

        cert_id = f"CERT-{int(ts)}-{batch_id[:8]}"

        # Seal = SHA256(cert_id + batch_id + score + grade + ledger_entry_hash)
        seal_payload = f"{cert_id}:{batch_id}:{score}:{grade}:{ledger_entry_hash}".encode()
        seal = hashlib.sha256(seal_payload).hexdigest()

        return TrustCertificate(
            certificate_id=cert_id,
            batch_id=batch_id,
            ai_readiness_score=score,
            grade=grade,
            is_certified=is_certified,
            decision=policy_decision.decision,
            dimension_scores=dim_scores,
            ledger_entry_hash=ledger_entry_hash,
            issued_at=ts,
            tamper_evident_seal=seal,
        )
