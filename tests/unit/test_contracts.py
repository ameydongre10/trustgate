"""Unit tests for Data Contracts: validation, breaking changes, ReDoS protection, and inference."""

import polars as pl
import pytest

from trustgate.contracts.inferrer import infer_contract
from trustgate.contracts.models import (
    ColumnContract,
    ColumnDType,
    DataContract,
    TrustLevel,
)
from trustgate.contracts.validator import (
    safe_regex_match,
    validate_contract,
)


@pytest.fixture
def sample_contract() -> DataContract:
    """Fixture providing a standard test DataContract."""
    return DataContract(
        name="customer_orders_contract",
        version="1.0.0",
        dataset_id="orders_stream",
        columns=[
            ColumnContract(name="order_id", dtype=ColumnDType.TEXT, required=True, nullable=False),
            ColumnContract(
                name="customer_age",
                dtype=ColumnDType.NUMERIC,
                required=True,
                min_value=18.0,
                max_value=120.0,
            ),
            ColumnContract(
                name="order_amount", dtype=ColumnDType.NUMERIC, required=True, min_value=0.0
            ),
            ColumnContract(
                name="status",
                dtype=ColumnDType.CATEGORICAL,
                allowed_values=["PENDING", "COMPLETED", "CANCELLED"],
            ),
            ColumnContract(
                name="coupon_code", dtype=ColumnDType.TEXT, required=False, nullable=True
            ),
        ],
    )


def test_contract_validation_success(sample_contract: DataContract) -> None:
    """Verify that a compliant batch validates successfully with no violations."""
    df = pl.DataFrame(
        {
            "order_id": ["ORD-101", "ORD-102"],
            "customer_age": [25.0, 45.0],
            "order_amount": [150.0, 20.5],
            "status": ["COMPLETED", "PENDING"],
            "coupon_code": ["SAVE10", None],
        }
    )

    report = validate_contract(df, sample_contract)
    assert report.is_valid is True
    assert not report.has_breaking_changes
    assert len(report.violations) == 0


def test_contract_breaking_missing_required_column(sample_contract: DataContract) -> None:
    """Verify that omitting a required column is flagged as BREAKING."""
    # Omit 'customer_age' (required)
    df = pl.DataFrame(
        {
            "order_id": ["ORD-101"],
            "order_amount": [150.0],
            "status": ["COMPLETED"],
        }
    )

    report = validate_contract(df, sample_contract)
    assert report.is_valid is False
    assert report.has_breaking_changes is True
    assert any("Required column 'customer_age' is missing" in bc for bc in report.breaking_changes)


def test_contract_nullability_and_range_violations(sample_contract: DataContract) -> None:
    """Verify nullability breach on non-nullable col, and min/max range breaches."""
    df = pl.DataFrame(
        {
            "order_id": ["ORD-101", None],  # Nullability violation on order_id
            "customer_age": [15.0, 150.0],  # Both violate min (18) and max (120)
            "order_amount": [-10.0, 50.0],  # Violates min_value 0.0
            "status": ["COMPLETED", "FRAUD"],  # "FRAUD" not in allowed set
        }
    )

    report = validate_contract(df, sample_contract)
    assert report.is_valid is False
    assert report.has_breaking_changes is True  # nullability breach is breaking

    v_types = [v.violation_type for v in report.violations]
    assert "NULLABILITY_VIOLATION" in v_types
    assert "RANGE_VIOLATION_MIN" in v_types
    assert "RANGE_VIOLATION_MAX" in v_types
    assert "ALLOWED_VALUES_VIOLATION" in v_types


def test_safe_regex_match_and_redos_guard() -> None:
    """Verify regex matching works and catastrophic backtracking patterns don't hang execution."""
    pattern = r"^[A-Z]{3}-\d{3}$"
    assert safe_regex_match(pattern, "ORD-123") is True
    assert safe_regex_match(pattern, "invalid") is False

    # Pathological ReDoS pattern: (a+)+$ on non-matching string "aaaa...b"
    redos_pattern = r"^(a+)+$"
    malicious_input = "a" * 30 + "b"

    # Must complete safely within time limit without freezing
    res = safe_regex_match(redos_pattern, malicious_input, timeout_seconds=0.01)
    # Result is a safe boolean without unhandled exceptions
    assert isinstance(res, bool)


def test_contract_inference() -> None:
    """Verify automated inference marks contract as INFERRED with correct types."""
    df = pl.DataFrame(
        {
            "user_id": [1, 2, 3],
            "balance": [10.5, 20.0, 50.25],
            "active": [True, False, True],
            "tier": ["SILVER", "GOLD", "PLATINUM"],
        }
    )

    contract = infer_contract(df, dataset_id="bank_users")
    assert contract.trust_level == TrustLevel.INFERRED
    assert contract.dataset_id == "bank_users"
    assert len(contract.columns) == 4

    balance_col = contract.get_column("balance")
    assert balance_col is not None
    assert balance_col.dtype == ColumnDType.NUMERIC
    assert balance_col.min_value == 10.5
    assert balance_col.max_value == 50.25
