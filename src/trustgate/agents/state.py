"""Typed execution state for the TrustGate agentic state machine."""

from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from trustgate.damage.predictor import DamagePrediction
from trustgate.pii.scanner import PIIScanReport
from trustgate.policy.engine import PolicyDecisionRecord
from trustgate.simulator.datasets import DatasetSpec
from trustgate.simulator.models import TargetModel


class TrustGateState(BaseModel):
    """Pydantic-typed state machine context passed across agent nodes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_id: str
    mode: Literal["labeled", "unlabeled"] = "unlabeled"
    raw_df: pl.DataFrame
    spec: DatasetSpec
    target_model: TargetModel
    ref_df: pl.DataFrame
    ref_stats: dict[str, Any]

    # Node outputs
    issue_profile: dict[str, Any] = Field(default_factory=dict)
    pii_report: PIIScanReport | None = None
    damage_prediction: DamagePrediction | None = None
    policy_decision: PolicyDecisionRecord | None = None
    repaired_df: pl.DataFrame | None = None
    explanation: str | None = None
    is_explanation_verified: bool = False

    # Workflow controls
    current_step: str = "START"
    requires_human_approval: bool = False
    human_approved: bool | None = None
    error: str | None = None
    failed_safe: bool = False
