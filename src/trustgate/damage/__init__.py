"""TrustGate Damage Prediction module with conformal uncertainty, TreeSHAP, and OOD guards."""

from trustgate.damage.baselines import (
    LinearDamageBaseline,
    MaxMetricDamageBaseline,
    RuleCountDamageBaseline,
)
from trustgate.damage.conformal import ConformalInterval, SplitConformalPredictor
from trustgate.damage.features import (
    DAMAGE_FEATURE_NAMES,
    DAMAGE_MONOTONIC_CONSTRAINTS,
    FEATURE_HUMAN_LABELS,
    build_feature_vector,
)
from trustgate.damage.ood_guard import OODCheckResult, OODGuard
from trustgate.damage.predictor import DamagePrediction, DamagePredictor
from trustgate.damage.registry import ModelRegistry
from trustgate.damage.validation import ProtocolMetrics, ValidationSuite

__all__ = [
    "DAMAGE_FEATURE_NAMES",
    "DAMAGE_MONOTONIC_CONSTRAINTS",
    "FEATURE_HUMAN_LABELS",
    "build_feature_vector",
    "ConformalInterval",
    "SplitConformalPredictor",
    "OODCheckResult",
    "OODGuard",
    "LinearDamageBaseline",
    "RuleCountDamageBaseline",
    "MaxMetricDamageBaseline",
    "DamagePrediction",
    "DamagePredictor",
    "ProtocolMetrics",
    "ValidationSuite",
    "ModelRegistry",
]
