"""Unit tests for PII detection, checksums, masking, HMAC tokenization, and erasure."""

import polars as pl

from trustgate.pii.checksums import (
    generate_verhoeff_checksum,
    validate_indian_ifsc,
    validate_indian_pan,
    validate_luhn,
    validate_verhoeff,
)
from trustgate.pii.detectors import PIIDetector
from trustgate.pii.erasure import ErasureRequest, execute_right_to_erasure
from trustgate.pii.masking import (
    mask_aadhaar,
    mask_credit_card,
    mask_email,
    mask_pan,
    mask_phone,
    tokenize_hmac,
)
from trustgate.pii.scanner import mask_dataframe, scan_dataframe_pii
from trustgate.utils.logging import redact_sensitive_text


def test_verhoeff_aadhaar_validation() -> None:
    """Verify Verhoeff checksum calculation and validation on Indian Aadhaar numbers."""
    # Generate valid 12-digit Aadhaar
    base11 = "21234567890"
    check_digit = generate_verhoeff_checksum(base11)
    valid_aadhaar = f"{base11}{check_digit}"

    assert validate_verhoeff(valid_aadhaar) is True

    # Mutate a single digit -> must fail
    invalid_aadhaar = valid_aadhaar[:-1] + str((check_digit + 1) % 10)
    assert validate_verhoeff(invalid_aadhaar) is False


def test_luhn_credit_card_validation() -> None:
    """Verify Luhn mod-10 algorithm for credit cards."""
    # Known test card (Visa)
    valid_visa = "4532015112830366"
    assert validate_luhn(valid_visa) is True

    # Single digit corruption
    invalid_card = "4532015112830367"
    assert validate_luhn(invalid_card) is False

    # Short/empty string
    assert validate_luhn("123") is False
    assert validate_luhn("0000000000000000") is False


def test_indian_pan_and_ifsc_validation() -> None:
    """Verify Indian PAN and IFSC structural validation."""
    # Valid individual PAN (4th char 'P' for Person)
    assert validate_indian_pan("ABCPE1234F") is True
    # Valid company PAN (4th char 'C' for Company)
    assert validate_indian_pan("AAACD1234C") is True
    # Invalid 4th char (Z is not a valid taxpayer entity)
    assert validate_indian_pan("ABCDZ1234F") is False
    # Invalid format
    assert validate_indian_pan("12345ABCDE") is False

    # Valid IFSC
    assert validate_indian_ifsc("HDFC0001234") is True
    assert validate_indian_ifsc("SBIN0000123") is True
    # 5th char must be 0
    assert validate_indian_ifsc("HDFC1001234") is False


def test_pii_detector_entities() -> None:
    """Verify detection across Aadhaar, PAN, CC, Email, UPI, and Phone."""
    detector = PIIDetector()

    # Generate a valid Aadhaar for the test (4-4-4 formatted)
    base11 = "59876543210"
    cd = generate_verhoeff_checksum(base11)
    full_12 = f"{base11}{cd}"
    valid_aadhaar = f"{full_12[:4]} {full_12[4:8]} {full_12[8:]}"

    text = f"""
    Customer profile:
    Email: priya.sharma@example.in
    Aadhaar: {valid_aadhaar}
    PAN: BLRPS1234K
    UPI: priya@okhdfcbank
    Mobile: +91 9876543210
    Card: 4532015112830366
    """

    findings = detector.detect_in_string(text)
    types_found = {f.entity_type for f in findings}

    assert "email" in types_found
    assert "indian_aadhaar" in types_found
    assert "indian_pan" in types_found
    assert "upi_id" in types_found
    assert "indian_phone" in types_found
    assert "credit_card" in types_found


def test_pii_masking_strategies() -> None:
    """Verify irreversible masking across types."""
    assert mask_email("rohit.verma@domain.com") == "r***a@domain.com"
    assert mask_credit_card("4532015112830366") == "****-****-****-0366"
    assert mask_aadhaar("9999 1234 5678") == "****-****-5678"
    assert mask_pan("ABCDE1234F") == "AB****34F"
    assert mask_phone("+919876543210") == "******210"


def test_salted_hmac_tokenization() -> None:
    """Verify HMAC tokenization properties: determinism with same salt, isolation with different salts."""
    val = "user_secret_identifier_123"
    token1 = tokenize_hmac(val, salt="salt_alpha")
    token2 = tokenize_hmac(val, salt="salt_alpha")
    token3 = tokenize_hmac(val, salt="salt_beta")

    assert token1.startswith("tok_hmac_")
    assert token1 == token2  # Deterministic for same salt
    assert token1 != token3  # Cryptographically isolated for different salts
    assert val not in token1


def test_dataframe_pii_scan_and_mask() -> None:
    """Verify end-to-end DataFrame PII scanning and masking."""
    base11 = "98765432101"
    cd = generate_verhoeff_checksum(base11)
    aadhaar_num = f"{base11}{cd}"

    df = pl.DataFrame(
        {
            "user_name": ["Alice", "Bob"],
            "email_address": ["alice@company.com", "bob@org.net"],
            "pan_number": ["ABCDE1234F", "XYZPA9999K"],
            "aadhaar": [aadhaar_num, aadhaar_num],
        }
    )

    scan_rep = scan_dataframe_pii(df)
    assert scan_rep.contains_unmasked_direct_pii is True
    assert scan_rep.has_critical_pii is True
    assert "pan_number" in scan_rep.pii_columns
    assert "aadhaar" in scan_rep.pii_columns

    masked_df = mask_dataframe(df, scan_rep, strategy="mask")
    # Verify raw PAN and Aadhaar are completely eradicated in the masked DataFrame
    for pan in ["ABCDE1234F", "XYZPA9999K"]:
        assert pan not in masked_df["pan_number"].to_list()
    assert aadhaar_num not in masked_df["aadhaar"].to_list()


def test_canary_string_leak_protection() -> None:
    """Canary test (Section 8.5): Verify canary PII is never exposed in logging."""
    canary_aadhaar = "9999 1234 5678"
    canary_pan = "ABCDE9999Z"
    canary_log = f"Processing batch with PAN {canary_pan} and UID {canary_aadhaar}"

    redacted_log = redact_sensitive_text(canary_log)
    assert canary_aadhaar not in redacted_log
    assert canary_pan not in redacted_log
    assert "[REDACTED_AADHAAR]" in redacted_log
    assert "[REDACTED_PAN]" in redacted_log


def test_right_to_erasure_tombstoning() -> None:
    """Verify GDPR / DPDP right to erasure tombstones records while generating receipt."""
    records = [
        {"id": 1, "email": "alice@test.com", "data": "clean"},
        {"id": 2, "email": "bob@subject.org", "data": "quarantined_payload"},
        {"id": 3, "email": "charlie@test.com", "data": "clean"},
    ]

    req = ErasureRequest(
        subject_identifier="bob@subject.org",
        dataset_id="user_stream",
        reason="User requested account deletion under GDPR Art 17",
    )

    sanitized, receipt = execute_right_to_erasure(req, records)

    assert receipt.rows_tombstoned_count == 1
    assert receipt.status == "TOMBSTONED"
    assert "bob@subject.org" not in str(sanitized)
    assert sanitized[1]["_tombstone"] is True
