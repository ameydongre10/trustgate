"""Pydantic data models for Batch and Column statistical profiling."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class OperatingMode(StrEnum):
    LABELED = "LABELED"
    UNLABELED = "UNLABELED"


class MetricStatus(StrEnum):
    OK = "OK"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    WARNING = "WARNING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class DriftMetricResult(BaseModel):
    test_name: str
    statistic: float
    p_value: float | None = None
    p_value_adjusted: float | None = None
    effect_size: float
    is_drifted: bool = False
    status: MetricStatus = MetricStatus.OK
    details: dict[str, Any] = Field(default_factory=dict)


class ColumnProfile(BaseModel):
    column_name: str
    inferred_type: str
    null_count: int
    null_rate: float
    distinct_count: int
    min_value: float | None = None
    max_value: float | None = None
    mean: float | None = None
    std: float | None = None
    zero_variance: bool = False
    outlier_count: int = 0
    outlier_rate: float = 0.0
    drift_metrics: dict[str, DriftMetricResult] = Field(default_factory=dict)


class QualitySummary(BaseModel):
    total_rows: int
    total_columns: int
    overall_null_rate: float
    duplicate_row_count: int
    duplicate_rate: float
    constant_columns: list[str] = Field(default_factory=list)
    high_missingness_columns: list[str] = Field(default_factory=list)


class LabelProfile(BaseModel):
    is_labeled: bool = False
    target_column: str | None = None
    class_balance: dict[str, float] = Field(default_factory=dict)
    num_classes: int = 0
    minority_class_rate: float | None = None
    label_shift_detected: bool = False
    label_shift_p_value: float | None = None
    estimated_noise_rate: float = 0.0
    status: MetricStatus = MetricStatus.NOT_APPLICABLE
    notes: str | None = None


class BatchProfile(BaseModel):
    batch_id: str
    operating_mode: OperatingMode
    quality_summary: QualitySummary
    column_profiles: dict[str, ColumnProfile]
    label_profile: LabelProfile
    max_psi: float = 0.0
    mean_psi: float = 0.0
    high_drift_columns: list[str] = Field(default_factory=list)
    profile_hash: str | None = None
