"""Unit tests for TrustGate core utilities."""

from pathlib import Path

import numpy as np
import pytest

from trustgate.utils.config import (
    PolicyConfig,
    ThresholdsConfig,
    load_policy_config,
    load_thresholds_config,
)
from trustgate.utils.errors import (
    DecompressionBombError,
    FailSafeAction,
)
from trustgate.utils.logging import redact_sensitive_text
from trustgate.utils.seeding import DeterministicContext, set_seed
from trustgate.utils.timing import Clock, Timer


def test_seeding_determinism() -> None:
    """Verify that set_seed ensures reproducible random draws."""
    set_seed(12345)
    r1 = np.random.rand(5)

    set_seed(12345)
    r2 = np.random.rand(5)

    np.testing.assert_allclose(r1, r2)


def test_deterministic_context() -> None:
    """Verify that DeterministicContext isolates seed without altering global state permanently."""
    set_seed(999)
    _ = np.random.rand(3)

    with DeterministicContext(seed=42):
        in_ctx_1 = np.random.rand(3)

    with DeterministicContext(seed=42):
        in_ctx_2 = np.random.rand(3)

    np.testing.assert_allclose(in_ctx_1, in_ctx_2)


def test_error_hierarchy() -> None:
    """Verify error serialization and default fail-safe actions."""
    err = DecompressionBombError("File ratio 120:1 exceeds limit 100:1", details={"ratio": 120})
    assert err.code == "ERR_DECOMPRESSION_BOMB"
    assert err.fail_safe_action == FailSafeAction.QUARANTINE

    d = err.to_dict()
    assert d["error_type"] == "DecompressionBombError"
    assert "Decompression bomb detected" in d["message"]
    assert d["details"]["ratio"] == 120


def test_pii_and_credential_redaction_in_logging() -> None:
    """Verify that sensitive tokens and direct PII are scrubbed in logs."""
    raw_msg = "User with Aadhaar 9999 1234 5678 and token api_key='secret-12345' submitted batch"
    redacted = redact_sensitive_text(raw_msg)

    assert "9999 1234 5678" not in redacted
    assert "[REDACTED_AADHAAR]" in redacted
    assert "secret-12345" not in redacted


def test_clock_and_timer() -> None:
    """Verify clock monotonic timing and test injection."""
    mock_clock = Clock(fixed_time=100.0)
    timer = Timer(clock=mock_clock)

    with timer:
        mock_clock.advance(2.5)

    assert pytest.approx(timer.elapsed) == 2.5
    assert pytest.approx(timer.elapsed_ms) == 2500.0


def test_config_loading() -> None:
    """Verify that policy and thresholds configuration YAML files load into validated models."""
    policy = load_policy_config(Path("configs/policy.yaml"))
    assert isinstance(policy, PolicyConfig)
    assert policy.thresholds.damage_approve_max == 0.05
    assert policy.hard_blocks.unmasked_direct_pii is True

    thresholds = load_thresholds_config(Path("configs/thresholds.yaml"))
    assert isinstance(thresholds, ThresholdsConfig)
    assert thresholds.drift.numeric_ks_pvalue_threshold == 0.05
    assert thresholds.quality.max_null_rate_alert == 0.20
