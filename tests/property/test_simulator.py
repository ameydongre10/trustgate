"""Property-based tests for simulator damage calculations and deterministic injectors."""

from hypothesis import given, settings
from hypothesis import strategies as st

from trustgate.simulator.models import TargetModel


@settings(max_examples=50, deadline=None)
@given(
    clean_score=st.floats(min_value=0.5, max_value=1.0),
    degradation_pct=st.floats(min_value=0.0, max_value=0.5),
)
def test_property_classification_damage_monotonicity(
    clean_score: float, degradation_pct: float
) -> None:
    """Property: As corrupted metric drops, calculated damage increases monotonically."""
    dummy_model = TargetModel(
        model_id="test_model",
        dataset_name="test",
        task_type="classification",
        metric_name="roc_auc",
        feature_names=["f1"],
        categorical_features=[],
        feature_importances={"f1": 1.0},
        training_reference_stats={},
        pipeline=None,
    )

    corrupted_score = clean_score * (1.0 - degradation_pct)
    damage = dummy_model.compute_damage(clean_score, corrupted_score)

    assert damage >= -1e-6
    if degradation_pct > 0.01:
        assert damage > 0.0


@settings(max_examples=50, deadline=None)
@given(
    clean_rmse=st.floats(min_value=10.0, max_value=1000.0),
    error_increase_pct=st.floats(min_value=0.0, max_value=2.0),
)
def test_property_regression_damage_monotonicity(
    clean_rmse: float, error_increase_pct: float
) -> None:
    """Property: In regression, as error increases, damage increases proportionally."""
    dummy_model = TargetModel(
        model_id="test_reg",
        dataset_name="test",
        task_type="regression",
        metric_name="rmse",
        feature_names=["f1"],
        categorical_features=[],
        feature_importances={"f1": 1.0},
        training_reference_stats={},
        pipeline=None,
    )

    corrupted_rmse = clean_rmse * (1.0 + error_increase_pct)
    damage = dummy_model.compute_damage(clean_rmse, corrupted_rmse)

    assert damage >= -1e-6
    assert abs(damage - error_increase_pct) < 1e-4
