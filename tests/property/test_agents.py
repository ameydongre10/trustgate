"""Property-based tests for agent guardrails and numeric claim verification."""

from hypothesis import given, settings
from hypothesis import strategies as st

from trustgate.agents.guardrails import GuardrailsEngine
from trustgate.agents.verifier import NumericClaimVerifier


@settings(max_examples=50, deadline=None)
@given(text=st.text(max_size=200))
def test_property_guardrails_no_control_characters(text: str) -> None:
    """Property: Sanitized text never contains dangerous ASCII control characters."""
    sanitized = GuardrailsEngine.sanitize_for_prompt(text)
    for ch in sanitized:
        code = ord(ch)
        if code < 32 and ch not in ("\n", "\t"):
            raise AssertionError(f"Dangerous control character found: code={code}")


@settings(max_examples=40, deadline=None)
@given(
    hallucinated_number=st.floats(min_value=500.0, max_value=99999.0, allow_nan=False),
)
def test_property_numeric_claim_verifier_catches_all_hallucinations(
    hallucinated_number: float,
) -> None:
    """Property: Any random large number not in context is flagged as a hallucination."""
    verifier = NumericClaimVerifier()
    ground_truth = {
        "batch_id": "B_01",
        "decision": "APPROVE",
        "risk_score": 10.0,
        "damage_upper_pct": 2.0,
        "pii_findings_count": 0,
        "cells_modified_count": 0,
        "top_drivers": [],
    }

    text = f"The model performance degraded by {hallucinated_number:.1f}% unexpectedly."
    is_valid, final_text, violations = verifier.verify_and_correct(text, ground_truth)

    assert not is_valid
    assert len(violations) > 0
    # Hallucinated number is never in the returned final text
    assert f"{hallucinated_number:.1f}" not in final_text
