"""Typed agent state machine workflow coordinating Profiler, Judge, and Explainer nodes."""

from trustgate.agents.explainer import explainer_node
from trustgate.agents.judge import judge_node
from trustgate.agents.llm_clients import LLMClient, TemplateLLMClient
from trustgate.agents.profiler import profiler_node
from trustgate.agents.state import TrustGateState
from trustgate.damage.predictor import DamagePredictor
from trustgate.policy.engine import Decision, PolicyDecisionRecord, PolicyEngine


class TrustGateWorkflow:
    """State machine DAG executing end-to-end data governance with fail-safe quarantine edges."""

    def __init__(
        self,
        policy_engine: PolicyEngine | None = None,
        damage_predictor: DamagePredictor | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.policy_engine = policy_engine or PolicyEngine()
        self.damage_predictor = damage_predictor
        self.llm_client = llm_client or TemplateLLMClient()

    def run(self, initial_state: TrustGateState) -> TrustGateState:
        """Execute state machine transitions with fail-safe exception containment."""
        state = initial_state
        state.current_step = "START"

        try:
            # Step 1: Profiler Agent Node
            state = profiler_node(state)
            if state.failed_safe:
                return self._fail_safe_terminate(state)

            # Step 2: Judge Agent Node (Deterministic Policy & Repair Planning)
            if self.damage_predictor is None:
                raise RuntimeError("DamagePredictor must be provided to TrustGateWorkflow.")

            state = judge_node(
                state=state,
                policy_engine=self.policy_engine,
                damage_predictor=self.damage_predictor,
            )
            if state.failed_safe:
                return self._fail_safe_terminate(state)

            # Step 3: Human-in-the-Loop Interrupt Check
            if state.requires_human_approval and state.human_approved is False:
                # Human rejected or timed out
                if state.policy_decision:
                    state.policy_decision.decision = Decision.QUARANTINE
                    state.policy_decision.rationale += (
                        " Quarantined after human operator rejection or timeout."
                    )

            # Step 4: Explainer Agent Node (Verified Narrative Generation)
            state = explainer_node(state, llm_client=self.llm_client)
            state.current_step = "END"

        except Exception as e:
            state.error = f"Unhandled state machine exception: {str(e)}"
            state.failed_safe = True
            return self._fail_safe_terminate(state)

        return state

    def _fail_safe_terminate(self, state: TrustGateState) -> TrustGateState:
        """Route catastrophic errors immediately to QUARANTINE (Rule 8: Fail-safe)."""
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
            hard_block_reasons=[state.error or "Unknown workflow failure"],
            rationale=f"Workflow fail-safe triggered: {state.error}. Batch quarantined automatically.",
            timestamp=0.0,
        )
        state = explainer_node(state, llm_client=self.llm_client)
        state.current_step = "TERMINATED_FAIL_SAFE"
        return state
