"""TrustGate Repair module with reference-fitted strategies, budget gating, and rollback verification."""

from trustgate.repair.planner import ClosedLoopRepairResult, RepairPlanner
from trustgate.repair.strategies import (
    RepairPatch,
    RepairResult,
    coerce_types,
    impute_missing_values,
    map_unseen_categories,
    realign_schema,
    repair_sentinels,
    winsorize_outliers,
)

__all__ = [
    "RepairPatch",
    "RepairResult",
    "impute_missing_values",
    "winsorize_outliers",
    "repair_sentinels",
    "coerce_types",
    "map_unseen_categories",
    "realign_schema",
    "ClosedLoopRepairResult",
    "RepairPlanner",
]
