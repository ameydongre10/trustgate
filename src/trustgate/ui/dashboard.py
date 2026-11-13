"""Streamlit Dashboard for TrustGate Impact-Aware Data Governance."""

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from trustgate.api.queue import JobQueue, JobRecord, JobStatus
from trustgate.audit.ledger import verify_audit_ledger
from trustgate.policy.engine import Decision
from trustgate.simulator.datasets import get_dataset
from trustgate.simulator.injectors import (
    inject_mcar,
    inject_mean_shift,
    inject_outliers_and_sentinels,
)


@st.cache_resource
def get_dashboard_queue() -> JobQueue:
    """Retrieve or initialize persistent dashboard JobQueue instance."""
    storage_dir = Path("trustgate_data")
    storage_dir.mkdir(parents=True, exist_ok=True)
    return JobQueue(storage_dir=storage_dir)


def render_dashboard() -> None:
    """Render main TrustGate multi-view dashboard."""
    st.set_page_config(
        page_title="TrustGate | Impact-Aware Data Governance",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    queue = get_dashboard_queue()

    st.sidebar.title("🛡️ TrustGate")
    st.sidebar.caption("Impact-Aware Data Trust & Verification Engine")

    nav_option = st.sidebar.radio(
        "Navigation",
        [
            "📊 Executive Overview",
            "🔍 Batch Inspector",
            "👥 Human Review Queue",
            "📜 Audit Ledger & Lineage",
            "⚡ Quick Batch Simulator",
        ],
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Target Downstream Model**")
    st.sidebar.info("Model: `credit_default_lgb`\n\nMetric: `ROC-AUC`\n\nThreshold: `5.0% Damage`")

    if nav_option == "📊 Executive Overview":
        render_executive_overview(queue)
    elif nav_option == "🔍 Batch Inspector":
        render_batch_inspector(queue)
    elif nav_option == "👥 Human Review Queue":
        render_human_review_queue(queue)
    elif nav_option == "📜 Audit Ledger & Lineage":
        render_audit_ledger(queue)
    elif nav_option == "⚡ Quick Batch Simulator":
        render_quick_simulator(queue)


def render_executive_overview(queue: JobQueue) -> None:
    """Render executive KPI cards and decision analytics."""
    st.header("Executive Governance Overview")
    st.caption("Real-time pipeline health, risk avoidance, and downstream model protection")

    jobs = queue.list_jobs()
    total_jobs = len(jobs)

    approved = sum(
        1
        for j in jobs
        if j.decision in (Decision.APPROVE.value, Decision.REPAIR_THEN_APPROVE.value)
    )
    repaired = sum(1 for j in jobs if j.decision == Decision.REPAIR_THEN_APPROVE.value)
    quarantined = sum(1 for j in jobs if j.decision == Decision.QUARANTINE.value)
    escalated = sum(
        1
        for j in jobs
        if j.status == JobStatus.ESCALATED_WAITING_FOR_REVIEW
        or j.decision == Decision.ESCALATE_TO_HUMAN.value
    )

    approval_rate = (approved / total_jobs * 100.0) if total_jobs > 0 else 0.0
    quarantine_rate = (quarantined / total_jobs * 100.0) if total_jobs > 0 else 0.0
    avg_risk = (queue.metrics_risk_score_sum / total_jobs) if total_jobs > 0 else 0.0
    damage_avoided = queue.metrics_damage_avoided_sum * 100.0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Batches Processed", total_jobs)
    col2.metric("Approval Rate", f"{approval_rate:.1f}%", delta=f"{approved} batches")
    col3.metric(
        "Quarantine Rate",
        f"{quarantine_rate:.1f}%",
        delta=f"-{quarantined} bad",
        delta_color="inverse",
    )
    col4.metric("Avg Risk Score", f"{avg_risk:.1f}/100")
    col5.metric("Model Damage Avoided", f"{damage_avoided:.1f}%", delta="Downstream Protected")

    st.divider()

    # Decision breakdown charts
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Decision Breakdown")
        dec_data = {
            "Decision": [
                "APPROVE",
                "REPAIR_THEN_APPROVE",
                "QUARANTINE",
                "ESCALATE_TO_HUMAN",
            ],
            "Count": [
                approved - repaired,
                repaired,
                quarantined,
                escalated,
            ],
        }
        df_dec = pd.DataFrame(dec_data)
        st.bar_chart(df_dec.set_index("Decision"))

    with right_col:
        st.subheader("Recent Governance Activity")
        if jobs:
            table_records = []
            for j in reversed(jobs[-8:]):
                table_records.append(
                    {
                        "Batch ID": j.batch_id,
                        "Status": j.status.value,
                        "Decision": j.decision or "N/A",
                        "Risk Score": f"{j.risk_score:.1f}" if j.risk_score is not None else "N/A",
                        "Damage Upper": (
                            f"{j.damage_upper * 100:.1f}%" if j.damage_upper is not None else "N/A"
                        ),
                        "AI Grade": j.ai_readiness_grade or "N/A",
                    }
                )
            st.dataframe(pd.DataFrame(table_records), use_container_width=True)
        else:
            st.info("No batches processed yet. Try running '⚡ Quick Batch Simulator' in sidebar!")


def render_batch_inspector(queue: JobQueue) -> None:
    """Detailed inspection of single batch: SHAP, damage bounds, and repair diffs."""
    st.header("Batch Inspector")

    jobs = queue.list_jobs()
    if not jobs:
        st.warning("No batches found in repository.")
        return

    job_options = {f"{j.batch_id} ({j.job_id}) - {j.decision or j.status.value}": j for j in jobs}
    selected_label = st.selectbox("Select Batch to Inspect", list(job_options.keys()))
    job: JobRecord = job_options[selected_label]

    # Badges row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Governance Decision**")
        color = (
            "green"
            if job.decision == "APPROVE"
            else "blue"
            if job.decision == "REPAIR_THEN_APPROVE"
            else "red"
        )
        st.markdown(f"### :{color}[{job.decision or 'PENDING'}]")
    with c2:
        st.markdown("**Risk Score**")
        st.markdown(f"### {job.risk_score:.1f}/100" if job.risk_score is not None else "### N/A")
    with c3:
        st.markdown("**Predicted Damage (90% Conformal)**")
        if job.damage_upper is not None and job.predicted_damage is not None:
            st.markdown(
                f"### ≤ {job.damage_upper * 100:.2f}% (Point: {job.predicted_damage * 100:.2f}%)"
            )
        else:
            st.markdown("### N/A")
    with c4:
        st.markdown("**AI-Readiness Grade**")
        st.markdown(f"### {job.ai_readiness_grade or 'N/A'}")

    st.divider()

    # SHAP feature drivers & Narrative Explanation
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Downstream Impact Drivers (SHAP)")
        if job.top_drivers:
            drivers_df = pd.DataFrame(job.top_drivers)
            drivers_df.columns = ["Issue Profile Factor", "SHAP Impact on Damage"]
            st.bar_chart(drivers_df.set_index("Issue Profile Factor"))
        else:
            st.info("No substantial damage drivers detected for this batch.")

    with col_right:
        st.subheader("Plain-Language Explanation")
        if job.explanation:
            st.markdown(job.explanation)
        else:
            st.info("Explanation pending generation.")

    st.divider()

    # Side-by-side Dirty vs Repaired preview
    st.subheader("Data Quality & Closed-Loop Repair Preview")
    if job.repaired_preview:
        st.success(f"Closed-loop repair applied to {job.repaired_cells_count} corrupted cells.")
        t1, t2 = st.tabs(["Raw / Submitted Batch", "Repaired Batch"])
        with t1:
            st.dataframe(pd.DataFrame(job.dirty_preview), use_container_width=True)
        with t2:
            st.dataframe(pd.DataFrame(job.repaired_preview), use_container_width=True)
    else:
        st.info("No repairs were needed or applied for this batch.")
        if job.dirty_preview:
            st.dataframe(pd.DataFrame(job.dirty_preview), use_container_width=True)


def render_human_review_queue(queue: JobQueue) -> None:
    """Human-in-the-loop review station for escalated batches."""
    st.header("Human Review & Escalation Queue")
    st.caption("Batches flagged with out-of-distribution profiles or borderline risk thresholds")

    escalated_jobs = [
        j
        for j in queue.list_jobs()
        if j.status == JobStatus.ESCALATED_WAITING_FOR_REVIEW
        or j.decision == Decision.ESCALATE_TO_HUMAN.value
    ]

    if not escalated_jobs:
        st.success("🎉 All clear! Zero batches currently waiting for human operator review.")
        return

    st.write(f"**{len(escalated_jobs)} Batch(es) Awaiting Operator Sign-off:**")

    for j in escalated_jobs:
        risk_val = j.risk_score or 0.0
        dmg_val = (j.damage_upper or 0.0) * 100.0
        with st.expander(
            f"⚠️ Batch {j.batch_id} (Risk: {risk_val:.1f}/100, Damage Upper: {dmg_val:.1f}%)",
            expanded=True,
        ):
            st.markdown(j.explanation or "No explanation generated.")

            st.write("---")
            action = st.radio(
                f"Action for Batch {j.batch_id}:",
                ["Approve Override (Proceed downstream)", "Confirm Quarantine (Reject batch)"],
                key=f"act_{j.job_id}",
            )
            comment = st.text_area(
                "Operator Review Comments / Audit Rationale:",
                value="Verified business context and accepted marginal drift profile.",
                key=f"comm_{j.job_id}",
            )

            if st.button("Submit Sign-off", key=f"btn_{j.job_id}"):
                approved = "Approve" in action
                queue.review_job(
                    job_id=j.job_id,
                    approved=approved,
                    comment=comment,
                    reviewer="operator_streamlit_ui",
                )
                st.success(f"Batch {j.batch_id} review committed to audit ledger!")
                time.sleep(0.5)
                st.rerun()


def render_audit_ledger(queue: JobQueue) -> None:
    """Tamper-evident ledger visualization and SHA-256 chain verification."""
    st.header("Tamper-Evident Audit Ledger")
    st.caption("Append-only SHA-256 cryptographic chain verifying complete governance lineage")

    valid, violations = verify_audit_ledger(queue.ledger_file)
    entries = queue.audit_ledger.entries

    c1, c2, c3 = st.columns(3)
    with c1:
        if valid:
            st.success("🔒 Hash Chain Integrity: VERIFIED")
        else:
            st.error(f"🚨 Integrity Check FAILED ({len(violations)} violations)!")
    with c2:
        st.metric("Total Ledger Entries", len(entries))
    with c3:
        head_short = (
            queue.audit_ledger.head_hash[:16] + "..." if queue.audit_ledger.head_hash else "GENESIS"
        )
        st.metric("Chain Head Hash", head_short)

    if violations:
        st.error("Audit Tampering Violations Detected:")
        for v in violations:
            st.write(f"- {v}")

    st.divider()

    st.subheader("Cryptographic Audit Chain Entries")
    if entries:
        records = []
        for e in reversed(entries):
            records.append(
                {
                    "Seq #": e.sequence_num,
                    "Batch ID": e.batch_id,
                    "Timestamp": e.timestamp,
                    "Decision / Action": e.payload.get("decision")
                    or e.payload.get("action", "EVENT"),
                    "Risk Score": e.payload.get("risk_score", "N/A"),
                    "Previous Hash": e.prev_hash[:12] + "...",
                    "Entry Hash": e.entry_hash[:12] + "...",
                }
            )
        st.dataframe(pd.DataFrame(records), use_container_width=True)
    else:
        st.info("Ledger is currently at Genesis state. Process a batch to write first block.")


def render_quick_simulator(queue: JobQueue) -> None:
    """Generate sample synthetic batches to test TrustGate live."""
    st.header("⚡ Quick Batch Simulator")
    st.caption(
        "Generate synthetic dirty batches to test TrustGate detection, damage prediction, and repair live"
    )

    _, test_df, spec, ref_stats = get_dataset(
        "credit_default", n_samples=150, seed=int(time.time())
    )

    corruption_type = st.selectbox(
        "Select Corruption Scenario:",
        [
            "Clean Data (Benchmark Baseline)",
            "High Null Rates in Key Columns (MCAR 25%)",
            "Severe Covariate Drift (Mean Shift +3.5 SD)",
            "Extreme Outlier Spikes (Income/Loan x100)",
        ],
    )

    if st.button("Generate & Process Batch"):
        with st.spinner("Injecting corruptions and running TrustGate Agent Workflow..."):
            corrupted_df = test_df
            if "MCAR" in corruption_type:
                corrupted_df, _ = inject_mcar(
                    corrupted_df, ["debt", "credit_score"], severity=0.25, seed=42
                )
            elif "Mean Shift" in corruption_type:
                corrupted_df, _ = inject_mean_shift(
                    corrupted_df, ["credit_score"], shift_stds=3.5, seed=42
                )
            elif "Outlier" in corruption_type:
                corrupted_df, _ = inject_outliers_and_sentinels(
                    corrupted_df, ["income", "debt"], severity=0.15, seed=42
                )

            # Convert to CSV bytes and submit
            csv_buf = corrupted_df.write_csv()
            job = queue.submit_batch(
                raw_bytes=csv_buf.encode("utf-8"),
                filename="simulated_batch.csv",
                target_model_name="credit_default_lgb",
            )

            # Wait briefly for execution
            for _ in range(30):
                if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                    break
                time.sleep(0.1)

            st.success(f"Batch {job.batch_id} processed!")
            r_score = job.risk_score or 0.0
            d_upper = (job.damage_upper or 0.0) * 100.0
            st.info(
                f"Verdict: **{job.decision}** | Risk Score: **{r_score:.1f}/100** | Damage Upper: **{d_upper:.1f}%**"
            )
            st.markdown(job.explanation or "")


if __name__ == "__main__":
    render_dashboard()
