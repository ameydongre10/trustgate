"""Unit tests for Phase 3 benchmark datasets, target models, corruptions, and experiment runner."""

from pathlib import Path

import polars as pl

from trustgate.simulator.compound import inject_compound_corruption
from trustgate.simulator.datasets import (
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
    inject_unit_error,
    inject_unseen_categories,
    inject_variance_scale,
)
from trustgate.simulator.models import train_target_model
from trustgate.simulator.runner import ExperimentRunner


def test_benchmark_datasets_generation() -> None:
    """Verify generation of all three benchmark datasets."""
    for d_name in ["credit_default", "telco_churn", "housing_prices"]:
        train_df, test_df, spec, ref_stats = get_dataset(d_name, n_samples=300, seed=42)

        assert train_df.height > 0
        assert test_df.height > 0
        assert spec.target_column in train_df.columns
        for f in spec.all_features:
            assert f in train_df.columns

        # Verify reference stats
        assert "numeric" in ref_stats
        assert "categorical" in ref_stats
        for nf in spec.numeric_features:
            assert nf in ref_stats["numeric"]
            assert "mean" in ref_stats["numeric"][nf]
            assert "quantiles" in ref_stats["numeric"][nf]


def test_target_model_training_and_evaluation() -> None:
    """Verify target model training, feature importance normalization, and damage calculation."""
    train_df, test_df, spec, ref_stats = get_dataset("credit_default", n_samples=1000, seed=42)

    # Train LightGBM model
    model = train_target_model(spec, train_df, ref_stats, model_type="lightgbm", seed=42)
    assert model.model_id == "credit_default_lightgbm"
    assert model.task_type == "classification"

    # Feature importances must sum to 1.0 (approx)
    total_imp = sum(model.feature_importances.values())
    assert abs(total_imp - 1.0) < 1e-4

    y_true = test_df[spec.target_column].to_numpy()
    clean_score = model.evaluate(test_df, y_true)
    assert clean_score >= 0.55

    # Damage computation
    assert model.compute_damage(clean_score, clean_score) == 0.0
    degraded_score = clean_score * 0.8
    damage = model.compute_damage(clean_score, degraded_score)
    assert 0.19 <= damage <= 0.21


def test_missingness_injectors() -> None:
    """Verify MCAR, MAR, and MNAR missingness injectors."""
    train_df, _, spec, _ = get_dataset("credit_default", n_samples=300, seed=42)

    # MCAR
    mcar_df, log_mcar = inject_mcar(train_df, columns=["income", "age"], severity=0.20, seed=42)
    assert mcar_df["income"].null_count() > 0
    assert mcar_df["age"].null_count() > 0
    assert log_mcar["affected_cells"] > 0

    # MAR (income nulls conditioned on age)
    mar_df, log_mar = inject_mar(
        train_df, target_col="income", condition_col="age", severity=0.30, seed=42
    )
    assert mar_df["income"].null_count() > 0
    assert log_mar["affected_cells"] > 0

    # MNAR (credit score nulls conditioned on high values)
    mnar_df, log_mnar = inject_mnar(train_df, target_col="credit_score", severity=0.30, seed=42)
    assert mnar_df["credit_score"].null_count() > 0
    assert log_mnar["affected_cells"] > 0


def test_covariate_drift_injectors() -> None:
    """Verify mean shift, variance scale, unit error, and unseen categories."""
    train_df, _, spec, _ = get_dataset("housing_prices", n_samples=300, seed=42)

    # Mean shift
    orig_mean = float(train_df["median_income"].mean())
    shifted_df, _ = inject_mean_shift(train_df, columns=["median_income"], shift_stds=2.0, seed=42)
    new_mean = float(shifted_df["median_income"].mean())
    assert new_mean > orig_mean

    # Variance scale
    orig_std = float(train_df["median_income"].std())
    scaled_df, _ = inject_variance_scale(
        train_df, columns=["median_income"], scale_factor=3.0, seed=42
    )
    new_std = float(scaled_df["median_income"].std())
    assert new_std > orig_std

    # Unit error
    unit_df, _ = inject_unit_error(train_df, column="median_income", multiplier=100.0, seed=42)
    assert abs(float(unit_df["median_income"].mean()) - (orig_mean * 100.0)) < 1.0

    # Unseen categories
    unseen_df, log_unseen = inject_unseen_categories(
        train_df, column="ocean_proximity", severity=0.25, seed=42
    )
    assert log_unseen["affected_cells"] > 0
    unique_cats = set(unseen_df["ocean_proximity"].unique())
    assert any("UNKNOWN" in str(c) or "NOVEL" in str(c) for c in unique_cats)


def test_concept_drift_and_label_noise() -> None:
    """Verify label noise and concept drift."""
    train_df, _, spec, _ = get_dataset("credit_default", n_samples=300, seed=42)

    # Symmetric label noise
    noisy_df, log_noise = inject_label_noise(train_df, target_col="default", severity=0.20, seed=42)
    assert log_noise["corrupted_labels"] > 0
    diff_count = (train_df["default"] != noisy_df["default"]).sum()
    assert diff_count > 0

    # Concept drift
    cd_df, log_cd = inject_concept_drift(
        train_df, target_col="default", condition_col="credit_score", severity=0.25, seed=42
    )
    assert log_cd["flipped_labels"] > 0


def test_duplicates_and_outliers() -> None:
    """Verify duplicate injection and outlier sentinel replacement."""
    train_df, _, spec, _ = get_dataset("telco_churn", n_samples=200, seed=42)

    # Duplicates
    dupe_df, log_dupe = inject_duplicates(train_df, duplicate_ratio=0.20, seed=42)
    expected_dupes = int(train_df.height * 0.20)
    assert dupe_df.height == train_df.height + expected_dupes
    assert log_dupe["n_duplicates"] == expected_dupes

    # Outliers / sentinels
    outlier_df, log_out = inject_outliers_and_sentinels(
        train_df, columns=["monthly_charges"], sentinel_values=[-999.0], severity=0.10, seed=42
    )
    assert log_out["affected_cells"] > 0
    assert -999.0 in outlier_df["monthly_charges"].to_list()


def test_compound_corruption() -> None:
    """Verify compound corruption applying multiple simultaneous issues."""
    train_df, _, spec, _ = get_dataset("credit_default", n_samples=300, seed=42)
    compound_df, logs = inject_compound_corruption(
        train_df, spec=spec, severity=0.20, num_issues=3, seed=42
    )

    assert len(logs) >= 2
    assert compound_df.height == train_df.height
    # Check that at least some changes occurred
    assert not compound_df.equals(train_df)


def test_experiment_runner(tmp_path: Path) -> None:
    """Verify ExperimentRunner runs simulations and logs Parquet + manifest."""
    runner = ExperimentRunner(output_dir=str(tmp_path), seed=42)
    parquet_path = runner.generate_experiment_dataset(n_rows_target=15, batch_size=150)

    assert parquet_path.exists()
    assert (tmp_path / "manifest.json").exists()

    df = pl.read_parquet(parquet_path)
    assert df.height >= 15
    assert "damage" in df.columns
    assert "importance_weighted_null_rate" in df.columns
    assert "importance_weighted_psi" in df.columns
    assert "model_id" in df.columns
