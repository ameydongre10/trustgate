"""Deterministic seeding utilities for reproducible execution."""

import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for deterministic execution across all supported libraries.

    Parameters
    ----------
    seed : int
        The integer seed to use. Default is 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class DeterministicContext:
    """Context manager for locally deterministic blocks with state restoration."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._random_state: tuple[Any, ...] | None = None
        self._np_state: Any = None

    def __enter__(self) -> "DeterministicContext":
        self._random_state = random.getstate()
        self._np_state = np.random.get_state()
        set_seed(self.seed)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._random_state is not None:
            random.setstate(self._random_state)
        if self._np_state is not None:
            np.random.set_state(self._np_state)
