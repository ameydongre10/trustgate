"""Monotonic clock interface and execution timing utilities."""

import time
from collections.abc import Callable
from typing import Any


class Clock:
    """Abstractable monotonic and UTC clock for deterministic time injection."""

    def __init__(self, fixed_time: float | None = None) -> None:
        self._fixed_time = fixed_time

    def time(self) -> float:
        """Return the current time in seconds since epoch."""
        if self._fixed_time is not None:
            return self._fixed_time
        return time.time()

    def monotonic(self) -> float:
        """Return monotonic clock value in seconds."""
        if self._fixed_time is not None:
            return self._fixed_time
        return time.monotonic()

    def advance(self, seconds: float) -> None:
        """Advance the fixed clock by a specified duration (testing utility)."""
        if self._fixed_time is not None:
            self._fixed_time += seconds


# Default global clock
system_clock = Clock()


class Timer:
    """Context manager for measuring elapsed execution time."""

    def __init__(self, clock: Clock | None = None) -> None:
        self.clock = clock or system_clock
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = self.clock.monotonic()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.end_time = self.clock.monotonic()
        self.elapsed = self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000.0


def time_it(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure and record execution time of a callable."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with Timer() as timer:
            result = func(*args, **kwargs)
        return result, timer.elapsed

    return wrapper
