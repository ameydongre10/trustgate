"""PII detectors with confidence scoring, checksum verification, and false-positive suppression."""

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from trustgate.pii.checksums import (
    validate_indian_pan,
    validate_luhn,
    validate_verhoeff,
)


class PIISeverity(StrEnum):
    CRITICAL = "CRITICAL"  # Direct government ID, full credit card, Aadhaar, PAN, SSN
    HIGH = "HIGH"  # Direct contact info: Phone, Email, UPI, Bank Account
    MEDIUM = "MEDIUM"  # Quasi-identifiers: IP address, DOB, GPS, Zip/PIN code
    LOW = "LOW"  # Broad demographic info: City, State, Country


class PIICategory(StrEnum):
    DIRECT_IDENTIFIER = "DIRECT_IDENTIFIER"
    FINANCIAL_IDENTIFIER = "FINANCIAL_IDENTIFIER"
    CONTACT_IDENTIFIER = "CONTACT_IDENTIFIER"
    QUASI_IDENTIFIER = "QUASI_IDENTIFIER"


class PIIFinding(BaseModel):
    entity_type: str
    raw_value: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: PIISeverity
    category: PIICategory
    column_name: str | None = None
    row_idx: int | None = None
    is_checksum_verified: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


# Regex patterns with strict non-digit boundary lookaround anchors
AADHAAR_REGEX = re.compile(r"(?<!\d)[2-9]{1}[0-9]{3}[-\s]?[0-9]{4}[-\s]?[0-9]{4}(?!\d)")
PAN_REGEX = re.compile(r"(?<![A-Za-z0-9])[A-Z]{5}[0-9]{4}[A-Z]{1}(?![A-Za-z0-9])")
CREDIT_CARD_REGEX = re.compile(
    r"(?<!\d)(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})(?!\d)"
)
INDIAN_PHONE_REGEX = re.compile(r"(?<!\d)(?:(?:\+91|91|0)[-\s]?)?([6-9]\d{9})(?!\d)")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
UPI_REGEX = re.compile(
    r"\b[a-zA-Z0-9.\-_]{2,256}@(okhdfcbank|okaxis|okicici|oksbi|paytm|ybl|ibl|upi|axl|apl|barodampay)\b"
)
IFSC_REGEX = re.compile(r"(?<![A-Za-z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Za-z0-9])")
SSN_REGEX = re.compile(r"(?<!\d)(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)")
IPV4_REGEX = re.compile(
    r"(?<!\d)(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?!\d)"
)

# Column names indicating direct PII
SUSPICIOUS_PII_COLUMN_NAMES = {
    "aadhaar": ("indian_aadhaar", PIISeverity.CRITICAL, PIICategory.DIRECT_IDENTIFIER),
    "pan": ("indian_pan", PIISeverity.CRITICAL, PIICategory.DIRECT_IDENTIFIER),
    "ssn": ("us_ssn", PIISeverity.CRITICAL, PIICategory.DIRECT_IDENTIFIER),
    "credit_card": ("credit_card", PIISeverity.CRITICAL, PIICategory.FINANCIAL_IDENTIFIER),
    "card_number": ("credit_card", PIISeverity.CRITICAL, PIICategory.FINANCIAL_IDENTIFIER),
    "phone": ("phone_number", PIISeverity.HIGH, PIICategory.CONTACT_IDENTIFIER),
    "mobile": ("phone_number", PIISeverity.HIGH, PIICategory.CONTACT_IDENTIFIER),
    "email": ("email", PIISeverity.HIGH, PIICategory.CONTACT_IDENTIFIER),
    "upi": ("upi_id", PIISeverity.HIGH, PIICategory.FINANCIAL_IDENTIFIER),
}


class PIIDetector:
    """Multi-identifier PII detector with algorithmic checksum verification."""

    def __init__(self, allow_list: set[str] | None = None) -> None:
        self.allow_list = {x.lower().strip() for x in (allow_list or set())}

    def detect_in_string(
        self,
        text: str,
        column_name: str | None = None,
        row_idx: int | None = None,
    ) -> list[PIIFinding]:
        """Scan a single text string for all supported PII entities."""
        if not text or not isinstance(text, str):
            return []

        clean_text = text.strip()
        if clean_text.lower() in self.allow_list:
            return []

        findings: list[PIIFinding] = []

        # 1. Indian Aadhaar Detection (12 digits with Verhoeff verification)
        for match in AADHAAR_REGEX.finditer(clean_text):
            candidate = match.group(0)
            digits = re.sub(r"\D", "", candidate)
            if len(digits) == 12 and validate_verhoeff(digits):
                findings.append(
                    PIIFinding(
                        entity_type="indian_aadhaar",
                        raw_value=candidate,
                        confidence=0.99,
                        severity=PIISeverity.CRITICAL,
                        category=PIICategory.DIRECT_IDENTIFIER,
                        column_name=column_name,
                        row_idx=row_idx,
                        is_checksum_verified=True,
                    )
                )

        # 2. Indian PAN Detection (Checked structure and 4th char)
        for match in PAN_REGEX.finditer(clean_text):
            candidate = match.group(0)
            if validate_indian_pan(candidate):
                findings.append(
                    PIIFinding(
                        entity_type="indian_pan",
                        raw_value=candidate,
                        confidence=0.95,
                        severity=PIISeverity.CRITICAL,
                        category=PIICategory.DIRECT_IDENTIFIER,
                        column_name=column_name,
                        row_idx=row_idx,
                        is_checksum_verified=True,
                    )
                )

        # 3. Credit Card Numbers (Luhn Algorithm)
        for match in CREDIT_CARD_REGEX.finditer(clean_text):
            candidate = match.group(0)
            digits = re.sub(r"\D", "", candidate)
            if validate_luhn(digits):
                findings.append(
                    PIIFinding(
                        entity_type="credit_card",
                        raw_value=candidate,
                        confidence=0.98,
                        severity=PIISeverity.CRITICAL,
                        category=PIICategory.FINANCIAL_IDENTIFIER,
                        column_name=column_name,
                        row_idx=row_idx,
                        is_checksum_verified=True,
                    )
                )

        # 4. Email Addresses
        for match in EMAIL_REGEX.finditer(clean_text):
            candidate = match.group(0)
            if candidate.lower() not in self.allow_list:
                findings.append(
                    PIIFinding(
                        entity_type="email",
                        raw_value=candidate,
                        confidence=0.95,
                        severity=PIISeverity.HIGH,
                        category=PIICategory.CONTACT_IDENTIFIER,
                        column_name=column_name,
                        row_idx=row_idx,
                    )
                )

        # 5. UPI ID Detection (VPA)
        for match in UPI_REGEX.finditer(clean_text):
            candidate = match.group(0)
            findings.append(
                PIIFinding(
                    entity_type="upi_id",
                    raw_value=candidate,
                    confidence=0.95,
                    severity=PIISeverity.HIGH,
                    category=PIICategory.FINANCIAL_IDENTIFIER,
                    column_name=column_name,
                    row_idx=row_idx,
                )
            )

        # 6. Indian Mobile Phone Numbers
        # Avoid double-counting numbers that are parts of credit cards or aadhaars
        existing_matches = {f.raw_value for f in findings}
        for match in INDIAN_PHONE_REGEX.finditer(clean_text):
            full_match = match.group(0)
            ten_digits = match.group(1)
            # Avoid if already part of an identified card or aadhaar
            if any(full_match in ex for ex in existing_matches):
                continue
            # Eliminate repeating numbers (e.g. 9999999999)
            if len(set(ten_digits)) >= 4:
                confidence = 0.90 if any(p in full_match for p in ("+91", "0")) else 0.75
                findings.append(
                    PIIFinding(
                        entity_type="indian_phone",
                        raw_value=full_match,
                        confidence=confidence,
                        severity=PIISeverity.HIGH,
                        category=PIICategory.CONTACT_IDENTIFIER,
                        column_name=column_name,
                        row_idx=row_idx,
                    )
                )

        # 7. US SSN
        for match in SSN_REGEX.finditer(clean_text):
            candidate = match.group(0)
            findings.append(
                PIIFinding(
                    entity_type="us_ssn",
                    raw_value=candidate,
                    confidence=0.95,
                    severity=PIISeverity.CRITICAL,
                    category=PIICategory.DIRECT_IDENTIFIER,
                    column_name=column_name,
                    row_idx=row_idx,
                )
            )

        # 8. IPv4 Address
        for match in IPV4_REGEX.finditer(clean_text):
            candidate = match.group(0)
            # Filter out localhost and broadcast
            if candidate not in ("127.0.0.1", "0.0.0.0", "255.255.255.255"):
                findings.append(
                    PIIFinding(
                        entity_type="ipv4_address",
                        raw_value=candidate,
                        confidence=0.85,
                        severity=PIISeverity.MEDIUM,
                        category=PIICategory.QUASI_IDENTIFIER,
                        column_name=column_name,
                        row_idx=row_idx,
                    )
                )

        return findings

    def check_column_name(self, col_name: str) -> tuple[str, PIISeverity, PIICategory] | None:
        """Check if column header itself explicitly indicates PII."""
        clean_col = re.sub(r"[^a-zA-Z0-9_]", "", col_name).lower()
        for key, (entity_type, sev, cat) in SUSPICIOUS_PII_COLUMN_NAMES.items():
            if key in clean_col:
                return entity_type, sev, cat
        return None
