"""DataFrame PII scanner and batch masking engine."""

import polars as pl
from pydantic import BaseModel, Field

from trustgate.pii.detectors import (
    PIIDetector,
    PIIFinding,
    PIISeverity,
)
from trustgate.pii.masking import mask_pii_value, tokenize_hmac


class PIIScanReport(BaseModel):
    contains_unmasked_direct_pii: bool = False
    has_critical_pii: bool = False
    has_high_pii: bool = False
    total_findings_count: int = 0
    pii_columns: dict[str, list[PIIFinding]] = Field(default_factory=dict)
    findings_by_severity: dict[str, int] = Field(default_factory=dict)
    findings_by_type: dict[str, int] = Field(default_factory=dict)
    quasi_identifier_risk_detected: bool = False
    details: list[str] = Field(default_factory=list)


def scan_dataframe_pii(
    df: pl.DataFrame,
    sample_size: int = 500,
    allow_list: set[str] | None = None,
) -> PIIScanReport:
    """Scan a DataFrame for PII across column headers and sampled data cells.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to inspect.
    sample_size : int
        Number of rows to sample per text column (default 500).
    allow_list : Optional[set[str]]
        Set of values or domains to ignore.

    Returns
    -------
    PIIScanReport
        Comprehensive scan findings and risk flags.
    """
    detector = PIIDetector(allow_list=allow_list)
    findings_by_col: dict[str, list[PIIFinding]] = {}
    severity_counts: dict[str, int] = {s.value: 0 for s in PIISeverity}
    type_counts: dict[str, int] = {}
    details: list[str] = []

    # 1. Header Analysis: Check if column names explicitly declare PII
    for col_name in df.columns:
        col_pii = detector.check_column_name(col_name)
        if col_pii:
            entity_type, sev, cat = col_pii
            finding = PIIFinding(
                entity_type=entity_type,
                raw_value=f"[HEADER: {col_name}]",
                confidence=0.90,
                severity=sev,
                category=cat,
                column_name=col_name,
                details={"source": "column_header_heuristic"},
            )
            findings_by_col.setdefault(col_name, []).append(finding)
            severity_counts[sev.value] += 1
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            details.append(f"Column '{col_name}' declared as {entity_type} ({sev.value})")

    # 2. Cell Content Analysis: Scan string columns
    n_rows = len(df)
    if n_rows > 0:
        actual_sample_size = min(n_rows, sample_size)
        sample_df = df.head(actual_sample_size)

        for col_name in df.columns:
            series = sample_df[col_name]
            # Check string/text or object columns
            if series.dtype in (pl.Utf8, pl.String, pl.Categorical):
                non_null = series.drop_nulls().cast(pl.Utf8).to_list()
                for row_idx, val in enumerate(non_null):
                    cell_findings = detector.detect_in_string(
                        str(val),
                        column_name=col_name,
                        row_idx=row_idx,
                    )
                    for f in cell_findings:
                        findings_by_col.setdefault(col_name, []).append(f)
                        severity_counts[f.severity.value] += 1
                        type_counts[f.entity_type] = type_counts.get(f.entity_type, 0) + 1

    # 3. Quasi-Identifier Re-identification Risk:
    # Detect if Date of Birth / Age is combined with Zip Code / PIN code
    col_names_lower = [c.lower() for c in df.columns]
    has_dob = any(
        x in c for c in col_names_lower for x in ("dob", "birth", "birth_date", "birthdate")
    )
    has_postal = any(
        x in c for c in col_names_lower for x in ("zip", "pincode", "pin_code", "postal")
    )

    quasi_risk = has_dob and has_postal
    if quasi_risk:
        details.append(
            "Quasi-identifier risk: Co-occurrence of DOB and Postal/PIN code facilitates re-identification."
        )

    total_findings = sum(len(f_list) for f_list in findings_by_col.values())
    has_critical = severity_counts.get(PIISeverity.CRITICAL.value, 0) > 0
    has_high = severity_counts.get(PIISeverity.HIGH.value, 0) > 0

    return PIIScanReport(
        contains_unmasked_direct_pii=has_critical or has_high,
        has_critical_pii=has_critical,
        has_high_pii=has_high,
        total_findings_count=total_findings,
        pii_columns=findings_by_col,
        findings_by_severity=severity_counts,
        findings_by_type=type_counts,
        quasi_identifier_risk_detected=quasi_risk,
        details=details,
    )


def mask_dataframe(
    df: pl.DataFrame,
    scan_report: PIIScanReport,
    strategy: str = "mask",  # "mask" (irreversible) or "tokenize" (HMAC)
    salt: str | None = None,
) -> pl.DataFrame:
    """Mask or tokenize all PII columns in a DataFrame based on scan findings.

    Parameters
    ----------
    df : pl.DataFrame
        Source DataFrame.
    scan_report : PIIScanReport
        Scan report identifying PII columns.
    strategy : str
        'mask' for irreversible asterisk masking, or 'tokenize' for salted HMAC.
    salt : Optional[str]
        Salt to use if tokenizing.

    Returns
    -------
    pl.DataFrame
        Sanitized DataFrame.
    """
    if not scan_report.pii_columns:
        return df

    result_df = df.clone()

    for col_name, findings in scan_report.pii_columns.items():
        if col_name not in result_df.columns:
            continue

        series = result_df[col_name]
        if series.dtype not in (pl.Utf8, pl.String):
            # Cast to string for masking
            series = series.cast(pl.Utf8)

        # Primary entity type in this column
        types = [f.entity_type for f in findings if not f.raw_value.startswith("[HEADER:")]
        dominant_type = max(set(types), key=types.count) if types else findings[0].entity_type

        # Apply transformation per cell
        def transform_val(val: str | None, entity_type: str = dominant_type) -> str | None:
            if val is None or val == "":
                return None
            if strategy == "tokenize":
                return tokenize_hmac(val, salt=salt)
            else:
                return mask_pii_value(val, entity_type=entity_type)

        transformed_list = [transform_val(v) for v in series.to_list()]
        result_df = result_df.with_columns(pl.Series(name=col_name, values=transformed_list))

    return result_df
