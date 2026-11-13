"""Property-based tests for statistical metrics using Hypothesis."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from trustgate.profiling.drift import calculate_psi


@settings(max_examples=50, deadline=None)
@given(
    ref=st.lists(
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=15,
        max_size=100,
    ),
    curr=st.lists(
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=15,
        max_size=100,
    ),
)
def test_property_psi_non_negative(ref: list[float], curr: list[float]) -> None:
    """Property: Population Stability Index (PSI) is mathematically non-negative (PSI >= 0.0)."""
    ref_arr = np.array(ref)
    curr_arr = np.array(curr)

    psi_val, _ = calculate_psi(ref_arr, curr_arr)
    assert not np.isnan(psi_val)
    assert psi_val >= 0.0


@settings(max_examples=25, deadline=None)
@given(
    values=st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=25,
        max_size=100,
    ),
)
def test_property_psi_self_identical_zero(values: list[float]) -> None:
    """Property: PSI of a distribution compared against itself is approximately 0.0."""
    arr = np.array(values)
    psi_val, _ = calculate_psi(arr, arr)
    assert psi_val == 0.0 or psi_val < 0.05
