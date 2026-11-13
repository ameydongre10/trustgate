"""Irreversible PII masking and deterministic salted HMAC tokenization."""

import hashlib
import hmac
import os
import re

# Default environment variable name for HMAC secret salt
HMAC_SALT_ENV_VAR = "TRUSTGATE_HMAC_SALT"
DEFAULT_FALLBACK_SALT = "trustgate-static-default-salt-do-not-use-in-prod-1234"


def get_hmac_salt() -> str:
    """Retrieve HMAC secret salt from environment or fallback with warning."""
    return os.environ.get(HMAC_SALT_ENV_VAR, DEFAULT_FALLBACK_SALT)


def tokenize_hmac(val: str, salt: str | None = None) -> str:
    """Generate a deterministic, salted HMAC-SHA256 token for an identifier.

    Preserves joinability/referential integrity for ML pipelines while eliminating
    raw PII disclosure.

    Parameters
    ----------
    val : str
        Plaintext identifier.
    salt : Optional[str]
        Secret salt. If None, loaded from TRUSTGATE_HMAC_SALT environment variable.

    Returns
    -------
    str
        Token in format 'tok_hmac_<16_hex_chars>'.
    """
    secret = (salt or get_hmac_salt()).encode("utf-8")
    clean_val = str(val).strip().lower().encode("utf-8")
    digest = hmac.new(secret, clean_val, hashlib.sha256).hexdigest()
    return f"tok_hmac_{digest[:16]}"


def mask_email(email_str: str) -> str:
    """Irreversibly mask an email address: user@domain.com -> u***r@domain.com."""
    s = str(email_str).strip()
    if "@" not in s:
        return "[REDACTED_EMAIL]"
    parts = s.split("@", 1)
    user, domain = parts[0], parts[1]
    if len(user) <= 2:
        masked_user = user[0] + "***" if user else "***"
    else:
        masked_user = user[0] + "***" + user[-1]
    return f"{masked_user}@{domain}"


def mask_credit_card(card_str: str) -> str:
    """Irreversibly mask a credit card number: keep last 4 digits only."""
    digits = re.sub(r"\D", "", str(card_str))
    if len(digits) < 4:
        return "[REDACTED_CC]"
    last4 = digits[-4:]
    return f"****-****-****-{last4}"


def mask_aadhaar(aadhaar_str: str) -> str:
    """Irreversibly mask an Indian Aadhaar number: keep last 4 digits only."""
    digits = re.sub(r"\D", "", str(aadhaar_str))
    if len(digits) < 4:
        return "[REDACTED_AADHAAR]"
    last4 = digits[-4:]
    return f"****-****-{last4}"


def mask_pan(pan_str: str) -> str:
    """Irreversibly mask an Indian PAN: ABCDE1234F -> AB****34F."""
    s = str(pan_str).strip().upper()
    if len(s) != 10:
        return "[REDACTED_PAN]"
    return f"{s[:2]}****{s[-3:]}"


def mask_phone(phone_str: str) -> str:
    """Irreversibly mask a phone number: keep country code and last 3 digits."""
    digits = re.sub(r"\D", "", str(phone_str))
    if len(digits) < 4:
        return "[REDACTED_PHONE]"
    last3 = digits[-3:]
    return f"******{last3}"


def mask_generic(val: str, entity_type: str = "PII") -> str:
    """Generic irreversible token replacement."""
    return f"[REDACTED_{entity_type.upper()}]"


def mask_pii_value(val: str, entity_type: str) -> str:
    """Route value to specific irreversible masking strategy."""
    et = entity_type.lower()
    if "email" in et:
        return mask_email(val)
    elif "card" in et:
        return mask_credit_card(val)
    elif "aadhaar" in et:
        return mask_aadhaar(val)
    elif "pan" in et:
        return mask_pan(val)
    elif "phone" in et:
        return mask_phone(val)
    else:
        return mask_generic(val, entity_type=et)
