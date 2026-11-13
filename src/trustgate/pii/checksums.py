"""Algorithmic checksum validators for Indian and international identifiers."""

import re

# =====================================================================
# 1. VERHOEFF ALGORITHM (Used for Indian Aadhaar 12-digit UID verification)
# =====================================================================
# The Verhoeff algorithm uses dihedral group D5 multiplication, permutation, and inverse tables.

VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def validate_verhoeff(number_str: str) -> bool:
    """Validate a numeric string using the Verhoeff checksum algorithm.

    Parameters
    ----------
    number_str : str
        Digits string (including the checksum digit as the last character).

    Returns
    -------
    bool
        True if checksum is valid, False otherwise.
    """
    clean_digits = re.sub(r"\D", "", str(number_str))
    if not clean_digits:
        return False

    c = 0
    reversed_digits = list(map(int, reversed(clean_digits)))
    for i, digit in enumerate(reversed_digits):
        p_val = VERHOEFF_P[i % 8][digit]
        c = VERHOEFF_D[c][p_val]

    return c == 0


def generate_verhoeff_checksum(number_str: str) -> int:
    """Generate the Verhoeff check digit for a 11-digit or n-digit base string."""
    clean_digits = re.sub(r"\D", "", str(number_str))
    c = 0
    reversed_digits = list(map(int, reversed(clean_digits)))
    for i, digit in enumerate(reversed_digits):
        p_val = VERHOEFF_P[(i + 1) % 8][digit]
        c = VERHOEFF_D[c][p_val]
    return VERHOEFF_INV[c]


# =====================================================================
# 2. LUHN ALGORITHM (Used for Credit Card number verification)
# =====================================================================


def validate_luhn(card_number_str: str) -> bool:
    """Validate a credit card number using the Luhn mod-10 algorithm.

    Parameters
    ----------
    card_number_str : str
        Credit card number string (digits, spaces, hyphens).

    Returns
    -------
    bool
        True if Luhn check passes and length is valid (13 to 19 digits).
    """
    clean_digits = re.sub(r"\D", "", str(card_number_str))
    if len(clean_digits) < 13 or len(clean_digits) > 19:
        return False

    # Check for all zeros
    if set(clean_digits) == {"0"}:
        return False

    digits = list(map(int, clean_digits))
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]

    total = sum(odd_digits)
    for d in even_digits:
        doubled = d * 2
        total += doubled if doubled < 10 else (doubled - 9)

    return total % 10 == 0


# =====================================================================
# 3. INDIAN PAN (Permanent Account Number) VALIDATOR
# =====================================================================

PAN_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$")
PAN_VALID_4TH_CHARS = {"P", "C", "H", "F", "A", "T", "B", "L", "J", "G"}


def validate_indian_pan(pan_str: str) -> bool:
    """Validate Indian Permanent Account Number (PAN) structure and holder type."""
    clean_pan = str(pan_str).strip().upper()
    if not PAN_REGEX.match(clean_pan):
        return False
    # 4th character designates taxpayer category (P=Individual, C=Company, etc.)
    return clean_pan[3] in PAN_VALID_4TH_CHARS


# =====================================================================
# 4. INDIAN IFSC (Indian Financial System Code) VALIDATOR
# =====================================================================

IFSC_REGEX = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")


def validate_indian_ifsc(ifsc_str: str) -> bool:
    """Validate Indian Financial System Code (IFSC) structure (4 letters, 0, 6 alphanumeric)."""
    return bool(IFSC_REGEX.match(str(ifsc_str).strip().upper()))
