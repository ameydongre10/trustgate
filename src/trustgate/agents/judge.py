"""Judge agent node executing deterministic policy rules and repair planning."""

from trustgate.agents.state import TrustGateState
from trustgate.damage.predictor import DamagePredictor
from trustgate.policy.engine import Decision, PolicyDecisionRecord, PolicyEngine


def judge_node(
    state: TrustGateState,
    policy_engine: PolicyEngine,
    damage_predictor: DamagePredictor,
) -> TrustGateState:
    """Evaluate batch against governance policy deterministically (Separation of Duties)."""
    if state.failed_safe:
        # State already flagged with an internal error: quarantine immediately
        state.policy_decision = PolicyDecisionRecord(
            batch_id=state.batch_id,
            decision=Decision.QUARANTINE,
            risk_score=100.0,
            predicted_damage=1.0,
            damage_upper=1.0,
            conformal_margin=0.0,
            applied_tier="standard",
            is_ood=False,
            requires_human_review=False,
            hard_block_triggered=True,
            hard_block_reasons=[state.error or "Internal system error occurred"],
            rationale=f"System fail-safe triggered: {state.error}. Batch quarantined automatically.",
            timestamp=0.0,
        )
        state.current_step = "JUDGED_FAIL_SAFE"
        return state

    try:
        from trustgate.simulator.runner import extract_fast_issue_features

        issue_profile = extract_fast_issue_features(
            state.raw_df, state.ref_df, state.target_model, state.spec
        )
        state.damage_prediction = damage_predictor.predict(issue_profile)

        decision_rec, repaired_df = policy_engine.evaluate(
            batch_id=state.batch_id,
            df=state.raw_df,
            spec=state.spec,
            target_model=state.target_model,
            ref_df=state.ref_df,
            ref_stats=state.ref_stats,
            damage_predictor=damage_predictor,
            pii_report=state.pii_report,
        )
        state.policy_decision = decision_rec
        state.repaired_df = repaired_df
        state.requires_human_approval = decision_rec.requires_human_review
        state.current_step = "JUDGED"
    except Exception as e:
        state.error = f"Judge node error: {str(e)}"
        state.failed_safe = True
        state.policy_decision = PolicyDecisionRecord(
            batch_id=state.batch_id,
            decision=Decision.QUARANTINE,
            risk_score=100.0,
            predicted_damage=1.0,
            damage_upper=1.0,
            conformal_margin=0.0,
            applied_tier="standard",
            is_ood=False,
            requires_human_review=False,
            hard_block_triggered=True,
            hard_block_reasons=[str(e)],
            rationale=f"Fail-safe quarantine triggered by Judge exception: {str(e)}",
            timestamp=0.0,
        )

    return state
