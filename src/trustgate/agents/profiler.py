"""Profiler agent node extracting statistical profiles, quality, drift, and PII."""

from trustgate.agents.state import TrustGateState
from trustgate.pii.scanner import scan_dataframe_pii
from trustgate.simulator.runner import extract_fast_issue_features


def profiler_node(state: TrustGateState) -> TrustGateState:
    """Run data profiling, drift detection, and PII scanning on raw batch."""
    try:
        # 1. Scan for PII (Aadhaar, PAN, CC, Email, Phone, UPI)
        pii_report = scan_dataframe_pii(state.raw_df)
        state.pii_report = pii_report

        # 2. Extract model-aware issue profile features
        issue_profile = extract_fast_issue_features(
            corrupted_df=state.raw_df,
            clean_ref_df=state.ref_df,
            target_model=state.target_model,
            spec=state.spec,
        )
        state.issue_profile = issue_profile
        state.current_step = "PROFILED"
    except Exception as e:
        state.error = f"Profiler node error: {str(e)}"
        state.failed_safe = True

    return state
