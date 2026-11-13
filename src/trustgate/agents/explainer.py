"""Explainer agent node formatting safe prompts and verifying all numerical claims."""

from trustgate.agents.guardrails import GuardrailsEngine
from trustgate.agents.llm_clients import LLMClient, TemplateLLMClient
from trustgate.agents.state import TrustGateState
from trustgate.agents.verifier import NumericClaimVerifier


def explainer_node(
    state: TrustGateState,
    llm_client: LLMClient | None = None,
    verifier: NumericClaimVerifier | None = None,
) -> TrustGateState:
    """Generate human-readable explanation and cryptographically verify all numerical claims."""
    client = llm_client or TemplateLLMClient()
    num_verifier = verifier or NumericClaimVerifier()

    if state.policy_decision is None:
        state.explanation = "No decision record available to explain."
        return state

    dec = state.policy_decision
    pii_count = state.pii_report.total_findings_count if state.pii_report else 0

    # 1. Format safe context with Zero Raw Data Cells (Rule 6)
    context = GuardrailsEngine.format_safe_summary_context(
        batch_id=state.batch_id,
        decision=dec.decision.value,
        risk_score=dec.risk_score,
        damage_upper=dec.damage_upper,
        top_factors=[],  # Aggregates only
        pii_count=pii_count,
        cells_modified=dec.cells_modified,
    )

    # 2. Generate raw narrative from LLM
    try:
        raw_narrative = client.generate_explanation(context)
    except Exception:
        # Fallback to offline template on LLM failure (Rule 5)
        raw_narrative = TemplateLLMClient().generate_explanation(context)

    # 3. Numeric Claim Verification (Rule 4: Every number is verified)
    is_valid, final_narrative, violations = num_verifier.verify_and_correct(
        generated_text=raw_narrative,
        ground_truth_context=context,
    )

    state.explanation = final_narrative
    state.is_explanation_verified = is_valid
    state.current_step = "EXPLAINED"

    return state
