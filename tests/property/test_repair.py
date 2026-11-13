"""Property-based tests for repair strategies and invariants using Hypothesis."""

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from trustgate.repair.strategies import impute_missing_values, winsorize_outliers


@settings(max_examples=40, deadline=None)
@given(
    values=st.lists(
        st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.none()),
        min_size=5,
        max_size=50,
    )
)
def test_property_imputation_leaves_zero_nulls(values: list[float | None]) -> None:
    """Property: After reference-fitted imputation, numeric column contains strictly 0 nulls."""
    df = pl.DataFrame({"feature": values})
    ref_stats = {"numeric": {"feature": {"quantiles": {"q_50": 42.0}, "mean": 42.0}}}

    repaired, patches = impute_missing_values(df, ref_stats)

    assert repaired["feature"].null_count() == 0
    # Number of patches equals original null count
    orig_nulls = sum(1 for v in values if v is None)
    assert len(patches) == orig_nulls


@settings(max_examples=40, deadline=None)
@given(
    values=st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False), min_size=5, max_size=50
    ),
    q01=st.floats(min_value=-50.0, max_value=0.0),
    q99=st.floats(min_value=10.0, max_value=50.0),
)
def test_property_winsorization_bounds(values: list[float], q01: float, q99: float) -> None:
    """Property: Winsorization strictly confines all numeric values to [q01, q99]."""
    df = pl.DataFrame({"metric": values})
    ref_stats = {"numeric": {"metric": {"quantiles": {"q_01": q01, "q_99": q99}}}}

    repaired, patches = winsorize_outliers(df, ref_stats)
    vals = repaired["metric"].to_numpy()

    assert np.all(vals >= q01 - 1e-5)
    assert np.all(vals <= q99 + 1e-5)
