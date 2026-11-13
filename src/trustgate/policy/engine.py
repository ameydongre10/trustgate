"""Deterministic policy evaluation engine enforcing thresholds, hard blocks, and separation of duties."""

from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from pydantic import BaseModel, Field

from trustgate.damage.predictor import DamagePrediction, DamagePredictor
from trustgate.pii.scanner import PIIScanReport
from trustgate.repair.planner import ClosedLoopRepairResult, RepairPlanner
from trustgate.simulator.datasets import DatasetSpec
from trustgate.simulator.models import TargetModel


class Decision(StrEnum):
    APPROVE = "APPROVE"
    REPAIR_THEN_APPROVE = "REPAIR_THEN_APPROVE"
    QUARANTINE = "QUARANTINE"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"


class PolicyDecisionRecord(BaseModel):
    batch_id: str
    decision: Decision
    risk_score: float = Field(..., ge=0.0, le=100.0)
    predicted_damage: float
    damage_upper: float
    conformal_margin: float
    applied_tier: str
    is_ood: bool
    requires_human_review: bool
    hard_block_triggered: bool
    hard_block_reasons: list[str] = Field(default_factory=list)
    repair_executed: bool = False
    repair_verdict: str | None = None
    cells_modified: int = 0
    rationale: str
    timestamp: float


class PolicyEngine:
    """Deterministic, policy-driven decision evaluator (Rule 4: The decision is never made by an LLM)."""

    def __init__(
        self,
        config_path: str | Path = "configs/policy.yaml",
        default_tier: str = "standard",
    ) -> None:
        self.config_path = Path(config_path)
        self.default_tier = default_tier
        self.config = self._load_config()
        self.repair_planner = RepairPlanner(
            max_budget_pct=float(
                self.config.get("thresholds", {}).get("max_repair_budget_pct", 0.15)
            )
        )

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {
                "thresholds": {
                    "damage_approve_max": 0.05,
                    "damage_repair_max": 0.20,
                    "damage_reject_min": 0.20,
                    "max_repair_budget_pct": 0.15,
                },
                "hard_blocks": {
                    "unmasked_direct_pii": True,
                    "breaking_schema_change": True,
                },
                "human_review": {
                    "escalate_on_ood": True,
                    "escalate_on_boundary_margin": 0.01,
                },
            }
        with open(self.config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_tier_thresholds(self, tier: str) -> dict[str, float]:
        """Resolve thresholds for specified risk tier with global fallbacks."""
        global_th = self.config.get("thresholds", {})
        tier_cfg = self.config.get("risk_tiers", {}).get(tier, {})

        approve_max = float(
            tier_cfg.get("damage_approve_max", global_th.get("damage_approve_max", 0.05))
        )
        repair_max = float(
            tier_cfg.get("damage_repair_max", global_th.get("damage_repair_max", 0.20))
        )
        reject_min = float(
            tier_cfg.get("damage_reject_min", global_th.get("damage_reject_min", 0.20))
        )

        return {
            "damage_approve_max": approve_max,
            "damage_repair_max": repair_max,
            "damage_reject_min": reject_min,
        }

    def compute_risk_score(
        self,
        damage_upper: float,
        pii_report: PIIScanReport | None,
        is_ood: bool,
        hard_block: bool,
    ) -> float:
        """Compute composite AI-risk score in [0.0, 100.0]."""
        if hard_block:
            return 100.0

        # Component 1: Model damage risk (scaled 0-50)
        damage_component = min(50.0, (damage_upper / 0.20) * 50.0)

        # Component 2: Privacy / PII risk (scaled 0-30)
        pii_component = 0.0
        if pii_report and pii_report.total_findings_count > 0:
            if pii_report.has_critical_pii or pii_report.contains_unmasked_direct_pii:
                pii_component = 30.0
            elif pii_report.has_high_pii:
                pii_component = 18.0
            else:
                pii_component = 8.0

        # Component 3: Distribution / OOD risk (scaled 0-20)
        ood_component = 20.0 if is_ood else 0.0

        total_risk = damage_component + pii_component + ood_component
        return float(min(100.0, max(0.0, round(total_risk, 1))))

    def evaluate(
        self,
        batch_id: str,
        df: pl.DataFrame,
        spec: DatasetSpec,
        target_model: TargetModel,
        ref_df: pl.DataFrame,
        ref_stats: dict[str, Any],
        damage_predictor: DamagePredictor,
        pii_report: PIIScanReport | None = None,
        schema_breaking: bool = False,
        tier: str | None = None,
        timestamp: float = 0.0,
    ) -> tuple[PolicyDecisionRecord, pl.DataFrame | None]:
        """Deterministically evaluate incoming batch against governance policies."""
        applied_tier = tier or self.default_tier
        th = self.get_tier_thresholds(applied_tier)
        hard_blocks_cfg = self.config.get("hard_blocks", {})
        human_cfg = self.config.get("human_review", {})

        hard_reasons = []

        # 1. Hard Block Check: Unmasked critical PII (Aadhaar, PAN, SSN, Credit Card)
        if hard_blocks_cfg.get("unmasked_direct_pii", True) and pii_report:
            if pii_report.has_critical_pii or pii_report.contains_unmasked_direct_pii:
                crit_count = pii_report.findings_by_severity.get(
                    "CRITICAL", pii_report.total_findings_count
                )
                hard_reasons.append(
                    f"Direct unmasked PII detected ({crit_count} critical entities)"
                )

        # 2. Hard Block Check: Breaking schema mutation
        if hard_blocks_cfg.get("breaking_schema_change", True) and schema_breaking:
            hard_reasons.append(
                "Breaking schema change detected (required columns dropped or incompatible type)"
            )

        if hard_reasons:
            risk_score = 100.0
            record = PolicyDecisionRecord(
                batch_id=batch_id,
                decision=Decision.QUARANTINE,
                risk_score=risk_score,
                predicted_damage=1.0,
                damage_upper=1.0,
                conformal_margin=0.0,
                applied_tier=applied_tier,
                is_ood=False,
                requires_human_review=False,
                hard_block_triggered=True,
                hard_block_reasons=hard_reasons,
                repair_executed=False,
                rationale=f"Hard governance block triggered: {'; '.join(hard_reasons)}. Batch quarantined immediately.",
                timestamp=timestamp,
            )
            return record, None

        # 3. Model Damage Prediction & Conformal Upper Bound
        from trustgate.simulator.runner import extract_fast_issue_features

        issue_profile = extract_fast_issue_features(df, ref_df, target_model, spec)
        damage_pred: DamagePrediction = damage_predictor.predict(issue_profile)
        damage_upper = damage_pred.damage_upper

        risk_score = self.compute_risk_score(
            damage_upper=damage_upper,
            pii_report=pii_report,
            is_ood=damage_pred.is_ood,
            hard_block=False,
        )

        # 4. OOD / Low-Confidence Human Escalation Check
        if human_cfg.get("escalate_on_ood", True) and damage_pred.is_ood:
            record = PolicyDecisionRecord(
                batch_id=batch_id,
                decision=Decision.ESCALATE_TO_HUMAN,
                risk_score=risk_score,
                predicted_damage=damage_pred.predicted_damage,
                damage_upper=damage_upper,
                conformal_margin=damage_pred.conformal_margin,
                applied_tier=applied_tier,
                is_ood=True,
                requires_human_review=True,
                hard_block_triggered=False,
                hard_block_reasons=[],
                rationale=(
                    f"Batch issue profile is out-of-distribution (confidence {damage_pred.confidence * 100:.1f}%). "
                    "Escalated to human operator per conservative fail-safe policy."
                ),
                timestamp=timestamp,
            )
            return record, None

        # 5. Policy Decision Tree Based on Predicted Damage
        # Case A: Damage Upper Bound <= Approval Threshold -> APPROVE
        if damage_upper <= th["damage_approve_max"]:
            # Boundary margin check
            boundary_margin = human_cfg.get("escalate_on_boundary_margin", 0.01)
            is_near_boundary = abs(damage_upper - th["damage_approve_max"]) <= boundary_margin

            if is_near_boundary and applied_tier == "critical":
                decision = Decision.ESCALATE_TO_HUMAN
                rationale = f"Predicted damage upper bound ({damage_upper * 100:.1f}%) is within boundary margin of critical approval threshold."
                requires_human = True
            else:
                decision = Decision.APPROVE
                rationale = f"Predicted damage upper bound {damage_upper * 100:.1f}% is safely within acceptable limit ({th['damage_approve_max'] * 100:.1f}%)."
                requires_human = False

            record = PolicyDecisionRecord(
                batch_id=batch_id,
                decision=decision,
                risk_score=risk_score,
                predicted_damage=damage_pred.predicted_damage,
                damage_upper=damage_upper,
                conformal_margin=damage_pred.conformal_margin,
                applied_tier=applied_tier,
                is_ood=False,
                requires_human_review=requires_human,
                hard_block_triggered=False,
                rationale=rationale,
                timestamp=timestamp,
            )
            return record, df

        # Case B: Damage Eligible for Closed-Loop Repair
        if damage_upper <= th["damage_repair_max"]:
            # Tier check: does critical tier require human escalation for repairs?
            tier_cfg = self.config.get("risk_tiers", {}).get(applied_tier, {})
            if tier_cfg.get("require_human_escalation_for_repairs", False):
                record = PolicyDecisionRecord(
                    batch_id=batch_id,
                    decision=Decision.ESCALATE_TO_HUMAN,
                    risk_score=risk_score,
                    predicted_damage=damage_pred.predicted_damage,
                    damage_upper=damage_upper,
                    conformal_margin=damage_pred.conformal_margin,
                    applied_tier=applied_tier,
                    is_ood=False,
                    requires_human_review=True,
                    hard_block_triggered=False,
                    rationale=f"Damage ({damage_upper * 100:.1f}%) is repairable but '{applied_tier}' policy mandates human escalation.",
                    timestamp=timestamp,
                )
                return record, None

            # Execute closed-loop repair
            repair_res: ClosedLoopRepairResult = self.repair_planner.execute_closed_loop_repair(
                df=df,
                spec=spec,
                target_model=target_model,
                ref_df=ref_df,
                ref_stats=ref_stats,
                damage_predictor=damage_predictor,
            )

            if repair_res.success and repair_res.verdict == "REPAIRED":
                # Re-check repaired damage against approval threshold
                repaired_damage = repair_res.post_damage
                repaired_risk = self.compute_risk_score(
                    damage_upper=repaired_damage,
                    pii_report=pii_report,
                    is_ood=False,
                    hard_block=False,
                )
                record = PolicyDecisionRecord(
                    batch_id=batch_id,
                    decision=Decision.REPAIR_THEN_APPROVE,
                    risk_score=repaired_risk,
                    predicted_damage=repaired_damage,
                    damage_upper=repaired_damage,
                    conformal_margin=damage_pred.conformal_margin,
                    applied_tier=applied_tier,
                    is_ood=False,
                    requires_human_review=False,
                    hard_block_triggered=False,
                    repair_executed=True,
                    repair_verdict=repair_res.verdict,
                    cells_modified=repair_res.cells_modified,
                    rationale=repair_res.message,
                    timestamp=timestamp,
                )
                return record, repair_res.repaired_df
            else:
                # Repair failed, budget exceeded, or degraded model -> Quarantine
                record = PolicyDecisionRecord(
                    batch_id=batch_id,
                    decision=Decision.QUARANTINE,
                    risk_score=risk_score,
                    predicted_damage=damage_pred.predicted_damage,
                    damage_upper=damage_upper,
                    conformal_margin=damage_pred.conformal_margin,
                    applied_tier=applied_tier,
                    is_ood=False,
                    requires_human_review=False,
                    hard_block_triggered=False,
                    repair_executed=True,
                    repair_verdict=repair_res.verdict,
                    cells_modified=repair_res.cells_modified,
                    rationale=f"Automated repair failed: {repair_res.message}. Batch quarantined.",
                    timestamp=timestamp,
                )
                return record, None

        # Case C: Damage Exceeds Repair Max -> Automatic Quarantine
        record = PolicyDecisionRecord(
            batch_id=batch_id,
            decision=Decision.QUARANTINE,
            risk_score=risk_score,
            predicted_damage=damage_pred.predicted_damage,
            damage_upper=damage_upper,
            conformal_margin=damage_pred.conformal_margin,
            applied_tier=applied_tier,
            is_ood=False,
            requires_human_review=False,
            hard_block_triggered=False,
            rationale=(
                f"Predicted damage upper bound ({damage_upper * 100:.1f}%) exceeds maximum allowable "
                f"rejection threshold ({th['damage_reject_min'] * 100:.1f}%)."
            ),
            timestamp=timestamp,
        )
        return record, None
