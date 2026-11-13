"""Property-based tests for deterministic seeding."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from trustgate.utils.seeding import set_seed


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_property_seeding_reproducibility(seed: int) -> None:
    """Property: For any arbitrary integer seed, resetting the seed yields bit-identical numpy arrays."""
    set_seed(seed)
    a = np.random.randn(20)

    set_seed(seed)
    b = np.random.randn(20)

    assert np.array_equal(a, b)
