"""Data Contract validator with breaking-change detection and ReDoS protection."""

import re
import time

import polars as pl
from pydantic import BaseModel, Field

from trustgate.contracts.models import (
    ColumnDType,
    DataContract,
)
from trustgate.ingest.schema import infer_series_type


class ViolationSeverity(str):
    BREAKING = "BREAKING"
    NON_BREAKING = "NON_BREAKING"
    WARNING = "WARNING"


class ColumnViolation(BaseModel):
    column_name: str
    violation_type: str
    severity: str
    message: str
    failed_rows_count: int = 0
    violation_rate: float = 0.0
    sample_values: list[str] = Field(default_factory=list)


class ContractValidationReport(BaseModel):
    is_valid: bool
    contract_name: str
    contract_version: str
    has_breaking_changes: bool
    breaking_changes: list[str] = Field(default_factory=list)
    non_breaking_changes: list[str] = Field(default_factory=list)
    violations: list[ColumnViolation] = Field(default_factory=list)
    total_rows_evaluated: int
    column_count: int


# Pattern to detect dangerous nested quantifiers causing exponential backtracking: (x+)+, (x*)*, etc.
REDOS_SUSPECT_PATTERN = re.compile(r"\([^)]*(\+|\*)[^)]*\)(\+|\*|\{\d+,?\})")


def is_vulnerable_to_redos(pattern_str: str) -> bool:
    """Detect common exponential backtracking / ReDoS structures in regex."""
    return bool(REDOS_SUSPECT_PATTERN.search(pattern_str))


def safe_regex_match(pattern_str: str, text: str, timeout_seconds: float = 0.05) -> bool:
    """Evaluate regex match with pre-validation against ReDoS structures and time limits."""
    if is_vulnerable_to_redos(pattern_str):
        # Reject catastrophic backtracking pattern immediately
        return False
    try:
        regex = re.compile(pattern_str)
        t0 = time.perf_counter()
        match = regex.match(text)
        if time.perf_counter() - t0 > timeout_seconds:
            return False
        return bool(match)
    except Exception:
        return False


def validate_contract(
    df: pl.DataFrame,
    contract: DataContract,
) -> ContractValidationReport:
    """Validate a DataFrame against a formal DataContract specification.

    Parameters
    ----------
    df : pl.DataFrame
        Batch DataFrame to validate.
    contract : DataContract
        Contract specification.

    Returns
    -------
    ContractValidationReport
        Detailed validation findings.
    """
    breaking_changes: list[str] = []
    non_breaking_changes: list[str] = []
    violations: list[ColumnViolation] = []

    batch_cols = {col.lower(): col for col in df.columns}
    contract_cols = {col.name.lower(): col for col in contract.columns}
    policy = contract.breaking_policy

    # 1. Check for missing columns
    for col_key, contract_col in contract_cols.items():
        if col_key not in batch_cols:
            if contract_col.required:
                msg = f"Required column '{contract_col.name}' is missing from batch"
                breaking_changes.append(msg)
                violations.append(
                    ColumnViolation(
                        column_name=contract_col.name,
                        violation_type="MISSING_REQUIRED_COLUMN",
                        severity=ViolationSeverity.BREAKING,
                        message=msg,
                        failed_rows_count=len(df),
                        violation_rate=1.0,
                    )
                )
            else:
                msg = f"Optional column '{contract_col.name}' is omitted"
                non_breaking_changes.append(msg)

    # 2. Check for unexpected / added columns
    for batch_col_key, original_name in batch_cols.items():
        if batch_col_key not in contract_cols:
            if policy.allow_added_optional_columns:
                non_breaking_changes.append(
                    f"Additional unexpected column '{original_name}' present in batch"
                )
            else:
                msg = f"Unexpected column '{original_name}' not permitted by contract policy"
                breaking_changes.append(msg)
                violations.append(
                    ColumnViolation(
                        column_name=original_name,
                        violation_type="UNEXPECTED_COLUMN",
                        severity=ViolationSeverity.BREAKING,
                        message=msg,
                    )
                )

    # 3. Column-level data constraints validation
    n_rows = len(df)
    for col_key, contract_col in contract_cols.items():
        if col_key not in batch_cols or n_rows == 0:
            continue

        actual_col_name = batch_cols[col_key]
        series = df[actual_col_name]

        # Check nullability
        null_count = series.null_count()
        if not contract_col.nullable and null_count > 0:
            msg = f"Column '{contract_col.name}' has {null_count} nulls but is defined non-nullable"
            violations.append(
                ColumnViolation(
                    column_name=contract_col.name,
                    violation_type="NULLABILITY_VIOLATION",
                    severity=ViolationSeverity.BREAKING,
                    message=msg,
                    failed_rows_count=null_count,
                    violation_rate=null_count / n_rows,
                )
            )

        # Check semantic dtype match
        inferred_type = infer_series_type(series)
        expected_type = contract_col.dtype.value
        if inferred_type != "null" and inferred_type != expected_type:
            # Allow categorical -> text or numeric float -> int widening if configured
            if expected_type == "text" and inferred_type == "categorical":
                pass
            else:
                msg = f"Type mismatch in '{contract_col.name}': expected {expected_type}, inferred {inferred_type}"
                severity = (
                    ViolationSeverity.BREAKING
                    if not policy.allow_widened_types
                    else ViolationSeverity.WARNING
                )
                violations.append(
                    ColumnViolation(
                        column_name=contract_col.name,
                        violation_type="TYPE_MISMATCH",
                        severity=severity,
                        message=msg,
                        violation_rate=1.0,
                    )
                )

        # Numeric Range Validation (min_value, max_value)
        if contract_col.dtype == ColumnDType.NUMERIC and (
            contract_col.min_value is not None or contract_col.max_value is not None
        ):
            non_null = series.drop_nulls()
            if len(non_null) > 0 and non_null.dtype.is_numeric():
                if contract_col.min_value is not None:
                    below_min = non_null.filter(non_null < contract_col.min_value)
                    if len(below_min) > 0:
                        msg = f"Column '{contract_col.name}' has {len(below_min)} values below min_value={contract_col.min_value}"
                        violations.append(
                            ColumnViolation(
                                column_name=contract_col.name,
                                violation_type="RANGE_VIOLATION_MIN",
                                severity=ViolationSeverity.NON_BREAKING,
                                message=msg,
                                failed_rows_count=len(below_min),
                                violation_rate=len(below_min) / n_rows,
                                sample_values=[str(v) for v in below_min.head(3).to_list()],
                            )
                        )

                if contract_col.max_value is not None:
                    above_max = non_null.filter(non_null > contract_col.max_value)
                    if len(above_max) > 0:
                        msg = f"Column '{contract_col.name}' has {len(above_max)} values above max_value={contract_col.max_value}"
                        violations.append(
                            ColumnViolation(
                                column_name=contract_col.name,
                                violation_type="RANGE_VIOLATION_MAX",
                                severity=ViolationSeverity.NON_BREAKING,
                                message=msg,
                                failed_rows_count=len(above_max),
                                violation_rate=len(above_max) / n_rows,
                                sample_values=[str(v) for v in above_max.head(3).to_list()],
                            )
                        )

        # Allowed Categorical Values (Enum check)
        if contract_col.allowed_values is not None:
            allowed_set = {str(x).lower() for x in contract_col.allowed_values}
            non_null = series.drop_nulls().cast(pl.Utf8)
            invalid_rows = non_null.filter(~non_null.str.to_lowercase().is_in(allowed_set))
            if len(invalid_rows) > 0:
                msg = f"Column '{contract_col.name}' has {len(invalid_rows)} values outside allowed set"
                violations.append(
                    ColumnViolation(
                        column_name=contract_col.name,
                        violation_type="ALLOWED_VALUES_VIOLATION",
                        severity=ViolationSeverity.NON_BREAKING,
                        message=msg,
                        failed_rows_count=len(invalid_rows),
                        violation_rate=len(invalid_rows) / n_rows,
                        sample_values=[str(v) for v in invalid_rows.unique().head(3).to_list()],
                    )
                )

        # Regex Pattern Check
        if contract_col.regex_pattern is not None:
            pattern = contract_col.regex_pattern
            non_null = series.drop_nulls().cast(pl.Utf8)
            # Evaluate on sample up to 500 rows with safe regex matcher
            sample_size = min(len(non_null), 500)
            sample_vals = non_null.head(sample_size).to_list()
            failed_samples = [v for v in sample_vals if not safe_regex_match(pattern, str(v))]
            if failed_samples:
                est_failed = int((len(failed_samples) / sample_size) * len(non_null))
                msg = f"Column '{contract_col.name}' failed regex pattern '{pattern}' on sample"
                violations.append(
                    ColumnViolation(
                        column_name=contract_col.name,
                        violation_type="REGEX_PATTERN_VIOLATION",
                        severity=ViolationSeverity.NON_BREAKING,
                        message=msg,
                        failed_rows_count=est_failed,
                        violation_rate=len(failed_samples) / sample_size,
                        sample_values=failed_samples[:3],
                    )
                )

    has_breaking = len(breaking_changes) > 0 or any(
        v.severity == ViolationSeverity.BREAKING for v in violations
    )
    is_valid = len(violations) == 0 and not has_breaking

    return ContractValidationReport(
        is_valid=is_valid,
        contract_name=contract.name,
        contract_version=contract.version,
        has_breaking_changes=has_breaking,
        breaking_changes=breaking_changes,
        non_breaking_changes=non_breaking_changes,
        violations=violations,
        total_rows_evaluated=n_rows,
        column_count=len(df.columns),
    )
