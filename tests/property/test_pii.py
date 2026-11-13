"""Property-based tests for PII masking and HMAC tokenization using Hypothesis."""

from hypothesis import given, settings
from hypothesis import strategies as st

from trustgate.pii.masking import (
    mask_credit_card,
    mask_email,
    tokenize_hmac,
)


@settings(max_examples=50, deadline=None)
@given(
    username=st.from_regex(r"[a-z]{4,12}", fullmatch=True),
    domain=st.from_regex(r"[a-z]{3,8}\.com", fullmatch=True),
)
def test_property_masked_email_never_contains_full_username(username: str, domain: str) -> None:
    """Property: The masked email must never contain the full plaintext username in the user portion."""
    email = f"{username}@{domain}"
    masked = mask_email(email)

    masked_user = masked.split("@")[0]
    assert username not in masked_user
    assert "@" in masked


@settings(max_examples=50, deadline=None)
@given(
    card_digits=st.from_regex(r"\d{16}", fullmatch=True),
)
def test_property_masked_card_never_contains_first_12_digits(card_digits: str) -> None:
    """Property: Masked credit card retains only the last 4 digits; first 12 are obscured."""
    masked = mask_credit_card(card_digits)
    first_12 = card_digits[:12]

    assert first_12 not in masked
    assert masked.endswith(card_digits[-4:])


@settings(max_examples=50, deadline=None)
@given(
    secret_val=st.text(min_size=1, max_size=100),
    salt=st.text(min_size=1, max_size=50),
)
def test_property_hmac_tokenization_properties(secret_val: str, salt: str) -> None:
    """Property: HMAC tokenization is strictly deterministic and irreversible."""
    tok1 = tokenize_hmac(secret_val, salt=salt)
    tok2 = tokenize_hmac(secret_val, salt=salt)

    assert tok1 == tok2
    assert tok1.startswith("tok_hmac_")
    assert len(tok1) == len("tok_hmac_") + 16
    if len(secret_val) > 4:
        assert secret_val not in tok1
