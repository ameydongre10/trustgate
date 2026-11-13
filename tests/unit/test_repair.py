"""Unit tests for Phase 5 repair strategies, idempotency, budget gating, and closed-loop rollback."""

import numpy as np
import polars as pl

from trustgate.damage.features import DAMAGE_FEATURE_NAMES
from trustgate.damage.predictor import DamagePredictor
from trustgate.repair.planner import RepairPlanner
from trustgate.repair.strategies import (
    coerce_types,
    impute_missing_values,
    map_unseen_categories,
    repair_sentinels,
    winsorize_outliers,
)
from trustgate.simulator.datasets import get_dataset
from trustgate.simulator.models import train_target_model


def test_reference_fitted_imputation() -> None:
    """Verify missing values are imputed strictly with reference statistics."""
    ref_stats = {
        "numeric": {
            "income": {"quantiles": {"q_50": 55000.0}, "mean": 54000.0},
        },
        "categorical": {
            "home_ownership": {"frequencies": {"MORTGAGE": 0.6, "RENT": 0.4}},
        },
    }

    df = pl.DataFrame(
        {
            "income": [60000.0, None, 80000.0],
            "home_ownership": ["RENT", None, "MORTGAGE"],
        }
    )

    repaired_df, patches = impute_missing_values(df, ref_stats)

    assert repaired_df["income"].null_count() == 0
    assert repaired_df["income"][1] == 55000.0
    assert repaired_df["home_ownership"][1] == "MORTGAGE"
    assert len(patches) == 2


def test_winsorize_and_sentinel_repair() -> None:
    """Verify outlier winsorization and sentinel value replacement."""
    ref_stats = {
        "numeric": {
            "val": {"quantiles": {"q_01": 10.0, "q_50": 50.0, "q_99": 100.0}},
        }
    }

    # Sentinel test
    df_sent = pl.DataFrame({"val": [50.0, -999.0, 9999.0, 60.0]})
    repaired_sent, p_sent = repair_sentinels(df_sent, ref_stats)
    assert repaired_sent["val"].to_list() == [50.0, 50.0, 50.0, 60.0]
    assert len(p_sent) == 2

    # Winsorize test
    df_out = pl.DataFrame({"val": [5.0, 50.0, 150.0]})
    repaired_out, p_out = winsorize_outliers(df_out, ref_stats)
    assert repaired_out["val"].to_list() == [10.0, 50.0, 100.0]
    assert len(p_out) == 2


def test_type_coercion_and_unseen_categories() -> None:
    """Verify messy string numbers are cleaned and unseen categories mapped."""
    ref_stats = {
        "numeric": {"price": {"quantiles": {"q_50": 100.0}}},
        "categorical": {"category": {"frequencies": {"A": 0.7, "B": 0.3}}},
    }

    df = pl.DataFrame(
        {
            "price": ["$1,250.00", "50%", "null", None],
            "category": ["A", "UNKNOWN_CATEGORY_X", "B", "NEW_TIER"],
        }
    )

    df_coerced, _ = coerce_types(df, ref_stats)
    assert df_coerced["price"].dtype == pl.Float64
    assert df_coerced["price"][0] == 1250.0
    assert df_coerced["price"][1] == 50.0

    df_cats, p_cats = map_unseen_categories(df_coerced, ref_stats)
    assert df_cats["category"].to_list() == ["A", "A", "B", "A"]
    assert len(p_cats) == 2


def test_repair_idempotency() -> None:
    """Verify that applying repair twice produces zero additional modifications."""
    ref_stats = {
        "numeric": {"val": {"quantiles": {"q_01": 10.0, "q_50": 50.0, "q_99": 100.0}}},
        "categorical": {"cat": {"frequencies": {"X": 0.8, "Y": 0.2}}},
    }

    df = pl.DataFrame(
        {
            "val": [5.0, None, 150.0, -999.0],
            "cat": ["X", None, "Z", "Y"],
        }
    )

    # Pass 1: Repairs applied
    df1, _ = impute_missing_values(df, ref_stats)
    df1, _ = repair_sentinels(df1, ref_stats)
    df1, _ = winsorize_outliers(df1, ref_stats)
    df1, _ = map_unseen_categories(df1, ref_stats)

    # Pass 2: Re-running repairs on clean data must yield 0 patches
    _, p2_imp = impute_missing_values(df1, ref_stats)
    _, p2_sen = repair_sentinels(df1, ref_stats)
    _, p2_win = winsorize_outliers(df1, ref_stats)
    _, p2_cat = map_unseen_categories(df1, ref_stats)

    assert len(p2_imp) == 0
    assert len(p2_sen) == 0
    assert len(p2_win) == 0
    assert len(p2_cat) == 0


def test_closed_loop_budget_cap_enforcement() -> None:
    """Verify planner rejects repair and rolls back if budget cap (15%) is exceeded."""
    train_df, test_df, spec, ref_stats = get_dataset("credit_default", n_samples=300, seed=42)
    model = train_target_model(spec, train_df, ref_stats, model_type="lightgbm", seed=42)

    # Train a minimal dummy damage predictor
    X = np.random.uniform(0, 0.3, size=(50, len(DAMAGE_FEATURE_NAMES)))
    y = np.random.uniform(0.01, 0.3, size=50)
    predictor = DamagePredictor(n_estimators=30, seed=42).fit(X, y)

    # Corrupt 80% of cells with nulls
    corrupted_df = test_df.clone()
    for col in spec.all_features:
        corrupted_df = corrupted_df.with_columns(
            pl.when(pl.Series(np.random.uniform(0, 1, size=test_df.height) < 0.80))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )

    planner = RepairPlanner(max_budget_pct=0.15)
    res = planner.execute_closed_loop_repair(
        df=corrupted_df,
        spec=spec,
        target_model=model,
        ref_df=train_df,
        ref_stats=ref_stats,
        damage_predictor=predictor,
    )

    assert not res.success
    assert res.budget_exceeded
    assert res.rolled_back
    assert res.verdict == "BUDGET_EXCEEDED"
    assert res.repaired_df.equals(corrupted_df)  # Strict rollback
