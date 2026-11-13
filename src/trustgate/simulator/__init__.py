"""TrustGate Simulator module for benchmark datasets, target models, corruptions, and experiment runner."""

from trustgate.simulator.compound import inject_compound_corruption
from trustgate.simulator.datasets import (
    DATASET_REGISTRY,
    DatasetSpec,
    extract_reference_stats,
    get_dataset,
)
from trustgate.simulator.injectors import (
    inject_concept_drift,
    inject_duplicates,
    inject_label_noise,
    inject_mar,
    inject_mcar,
    inject_mean_shift,
    inject_mnar,
    inject_outliers_and_sentinels,
    inject_schema_mutation,
    inject_type_corruption,
    inject_unit_error,
    inject_unseen_categories,
    inject_variance_scale,
)
from trustgate.simulator.models import TargetModel, train_target_model
from trustgate.simulator.runner import ExperimentRunner, extract_fast_issue_features

__all__ = [
    "DATASET_REGISTRY",
    "DatasetSpec",
    "get_dataset",
    "extract_reference_stats",
    "TargetModel",
    "train_target_model",
    "inject_mcar",
    "inject_mar",
    "inject_mnar",
    "inject_mean_shift",
    "inject_variance_scale",
    "inject_unit_error",
    "inject_unseen_categories",
    "inject_concept_drift",
    "inject_label_noise",
    "inject_duplicates",
    "inject_outliers_and_sentinels",
    "inject_type_corruption",
    "inject_schema_mutation",
    "inject_compound_corruption",
    "ExperimentRunner",
    "extract_fast_issue_features",
]
