"""Pydantic-based configuration management and YAML parsing."""

from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, Field


class PolicyThresholds(BaseModel):
    damage_approve_max: float = Field(0.05, ge=0.0, le=1.0)
    damage_repair_max: float = Field(0.20, ge=0.0, le=1.0)
    damage_reject_min: float = Field(0.20, ge=0.0, le=1.0)
    ood_anomaly_threshold: float = Field(-0.5)
    conformal_confidence: float = Field(0.90, gt=0.5, lt=1.0)
    max_repair_budget_pct: float = Field(0.15, ge=0.0, le=1.0)


class HardBlockRules(BaseModel):
    unmasked_direct_pii: bool = True
    breaking_schema_change: bool = True
    corrupted_audit_hash: bool = True


class RiskTierConfig(BaseModel):
    damage_approve_max: float
    damage_repair_max: float
    conformal_confidence: float = 0.90
    require_human_escalation_for_repairs: bool = False


class HumanReviewConfig(BaseModel):
    escalate_on_ood: bool = True
    escalate_on_boundary_margin: float = 0.01
    sla_hours: int = 24
    default_action_on_sla_timeout: str = "QUARANTINE"


class PolicyConfig(BaseModel):
    version: str = "1.0.0"
    default_risk_tier: str = "standard"
    thresholds: PolicyThresholds = Field(default_factory=PolicyThresholds)
    hard_blocks: HardBlockRules = Field(default_factory=HardBlockRules)
    risk_tiers: dict[str, RiskTierConfig] = Field(default_factory=dict)
    human_review: HumanReviewConfig = Field(default_factory=HumanReviewConfig)


class DriftThresholds(BaseModel):
    numeric_ks_pvalue_threshold: float = 0.05
    numeric_psi_warn_threshold: float = 0.10
    numeric_psi_alert_threshold: float = 0.25
    categorical_chi2_pvalue_threshold: float = 0.05
    categorical_cramers_v_threshold: float = 0.15
    js_divergence_threshold: float = 0.10
    fdr_alpha: float = 0.05
    min_sample_size_for_hypothesis_tests: int = 30


class QualityThresholds(BaseModel):
    max_null_rate_warn: float = 0.05
    max_null_rate_alert: float = 0.20
    max_duplicate_rate_warn: float = 0.01
    max_duplicate_rate_alert: float = 0.05
    z_score_outlier_threshold: float = 3.5


class ThresholdsConfig(BaseModel):
    drift: DriftThresholds = Field(default_factory=DriftThresholds)
    quality: QualityThresholds = Field(default_factory=QualityThresholds)


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Safely load and parse a YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_policy_config(path: Path | str | None = None) -> PolicyConfig:
    """Load PolicyConfig from a file or default config directory."""
    target_path = Path(path) if path else Path("configs/policy.yaml")
    data = load_yaml(target_path)
    return cast(PolicyConfig, PolicyConfig.model_validate(data))


def load_thresholds_config(path: Path | str | None = None) -> ThresholdsConfig:
    """Load ThresholdsConfig from a file or default config directory."""
    target_path = Path(path) if path else Path("configs/thresholds.yaml")
    data = load_yaml(target_path)
    return cast(ThresholdsConfig, ThresholdsConfig.model_validate(data))
