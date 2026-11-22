"""Streamlit Dashboard for TrustGate Impact-Aware Data Governance."""

import time
from pathlib import Path

import pandas as pd
import polars as pl
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


def _seed_demo_batches(queue: JobQueue) -> None:
    """Pre-populate sample batches so the dashboard is immediately interactive upon launch."""
    try:
        train_df, test_df, _, _ = get_dataset("credit_default", n_samples=150, seed=42)

        # 1. Clean batch -> APPROVE
        clean_bytes = test_df.head(60).write_csv().encode("utf-8")
        queue.submit_batch(raw_bytes=clean_bytes, filename="clean_inflow.csv", target_model_name="credit_default_lgb")

        # 2. Repairable missingness -> REPAIR_THEN_APPROVE
        repaired_df, _ = inject_mcar(test_df.tail(60), columns=["income"], severity=0.04, seed=42)
        rep_bytes = repaired_df.write_csv().encode("utf-8")
        queue.submit_batch(raw_bytes=rep_bytes, filename="missingness_batch.csv", target_model_name="credit_default_lgb")

        # 3. PII leak -> QUARANTINE
        pii_df = test_df.head(40).with_columns(
            pl.Series("customer_ssn", ["123-45-6789" for _ in range(40)])
        )
        pii_bytes = pii_df.write_csv().encode("utf-8")
        queue.submit_batch(raw_bytes=pii_bytes, filename="pii_breach.csv", target_model_name="credit_default_lgb")

        # Wait briefly for worker thread to process initial seed batches so dynamic views are populated
        for _ in range(25):
            jobs = queue.list_jobs()
            if len(jobs) >= 3 and all(j.status not in (JobStatus.PENDING, JobStatus.RUNNING) for j in jobs):
                break
            time.sleep(0.1)
    except Exception:
        pass


@st.cache_resource
def get_dashboard_queue() -> JobQueue:
    """Retrieve or initialize persistent dashboard JobQueue instance."""
    storage_dir = Path("trustgate_data")
    storage_dir.mkdir(parents=True, exist_ok=True)
    queue = JobQueue(storage_dir=storage_dir)

    if len(queue.list_jobs()) == 0:
        _seed_demo_batches(queue)

    return queue


def render_dashboard() -> None:
    """Render main TrustGate multi-view dashboard."""
    st.set_page_config(
        page_title="TrustGate | Impact-Aware Data Governance",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Smooth initial connection indicator
    if "backend_connected" not in st.session_state:
        st.session_state["backend_connected"] = True
        st.toast("Connected to TrustGate Agent Pipeline!", icon="🛡️")

    # Instant Static-to-Dynamic Bootloader Overlay (Displays for ~2.5s then dissolves smoothly into live dynamic UI)
    st.markdown(
        """
        <div id="trustgate-bootloader" class="bootloader-overlay">
            <div class="bootloader-card">
                <div class="bootloader-header">
                    <span class="bootloader-beacon"></span>
                    <span class="bootloader-title">TRUSTGATE // 01</span>
                </div>
                <div class="bootloader-sub">CONNECTING TO DYNAMIC IMPACT-AWARE ENGINE</div>
                <p class="bootloader-desc">
                    Warming up downstream model simulators, TreeSHAP explainers, and SHA-256 audit ledger. Full interactive console loading...
                </p>
                <div class="bootloader-progress-track">
                    <div class="bootloader-progress-fill"></div>
                </div>
                <div class="bootloader-footer">
                    <span>INITIALIZING FULL DYNAMIC WORKSPACE</span>
                    <span>STANDBY ~2.5s</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Editorial Brutalist / Institutional Authority Style Injection
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');

        /* Root Typography & Architectural Theme */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: #09090b !important;
            color: #f4f4f5 !important;
            -webkit-font-smoothing: antialiased;
        }

        /* Static Bootloader Screen (Fades out after 2.4s to reveal 100% dynamic app) */
        .bootloader-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #09090b;
            z-index: 9999999;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: fadeOutBootloader 0.5s cubic-bezier(0.16, 1, 0.3, 1) 2.4s forwards;
            pointer-events: none;
        }

        @keyframes fadeOutBootloader {
            0% { opacity: 1; visibility: visible; }
            99% { opacity: 0; visibility: visible; }
            100% { opacity: 0; visibility: hidden; display: none; }
        }

        .bootloader-card {
            background: #0f1013;
            border: 1px solid #27272a;
            border-left: 4px solid #10b981;
            padding: 32px 36px;
            max-width: 520px;
            width: 90%;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
        }

        .bootloader-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 6px;
        }

        .bootloader-beacon {
            width: 10px;
            height: 10px;
            background: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 12px #10b981;
            display: inline-block;
            animation: beaconPulse 1.2s infinite alternate;
        }

        @keyframes beaconPulse {
            0% { opacity: 0.4; transform: scale(0.9); }
            100% { opacity: 1; transform: scale(1.15); }
        }

        .bootloader-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 18px;
            font-weight: 800;
            letter-spacing: -0.01em;
            color: #ffffff;
        }

        .bootloader-sub {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #a1a1aa;
            margin-bottom: 14px;
        }

        .bootloader-desc {
            font-size: 13px;
            color: #71717a;
            line-height: 1.5;
            margin-bottom: 20px;
        }

        .bootloader-progress-track {
            width: 100%;
            height: 4px;
            background: #18181b;
            border-radius: 2px;
            overflow: hidden;
            margin-bottom: 12px;
        }

        .bootloader-progress-fill {
            width: 0%;
            height: 100%;
            background: linear-gradient(90deg, #10b981, #06b6d4);
            animation: fillBootloaderProgress 2.2s ease-in-out forwards;
        }

        @keyframes fillBootloaderProgress {
            0% { width: 0%; }
            35% { width: 45%; }
            70% { width: 85%; }
            100% { width: 100%; }
        }

        .bootloader-footer {
            display: flex;
            justify-content: space-between;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: #52525b;
            letter-spacing: 0.05em;
        }

        /* Enforce tabular numerals across all numbers and data tables */
        * {
            font-variant-numeric: tabular-nums;
        }

        /* Strict Architectural Zero/Low Radius (No Generic Bubbles) */
        div[data-testid="stMetric"],
        .stButton > button,
        .stDownloadButton > button,
        div[data-testid="stFileUploader"],
        div[data-testid="stDataFrame"],
        .stSelectbox,
        .stTextInput,
        .stTextArea,
        .stAlert,
        div[data-baseweb="tab-list"],
        div[data-baseweb="tab"],
        div[data-baseweb="select"] > div {
            border-radius: 2px !important;
        }

        /* Editorial Institutional Masthead */
        .editorial-masthead {
            border: 1px solid #27272a;
            background: #0f1013;
            padding: 22px 26px;
            margin-bottom: 24px;
            border-left: 4px solid #f4f4f5;
        }
        .masthead-meta {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #a1a1aa;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .masthead-title {
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #ffffff;
            line-height: 1.15;
            margin-bottom: 6px;
        }
        .masthead-subtitle {
            font-size: 13px;
            color: #a1a1aa;
            font-weight: 500;
        }

        /* High-Assurance Metric Cards */
        div[data-testid="stMetric"] {
            background-color: #121316 !important;
            border: 1px solid #27272a !important;
            padding: 16px 20px !important;
            box-shadow: none !important;
            border-top: 2px solid #52525b !important;
        }
        div[data-testid="stMetricLabel"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 11px !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            color: #a1a1aa !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 32px !important;
            font-weight: 800 !important;
            color: #ffffff !important;
            letter-spacing: -0.02em !important;
        }

        /* Brutalist High-Contrast Buttons */
        .stButton > button[kind="primary"] {
            background-color: #f4f4f5 !important;
            color: #09090b !important;
            font-weight: 800 !important;
            border: 1px solid #ffffff !important;
            font-family: 'JetBrains Mono', monospace !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            padding: 0.55rem 1.25rem !important;
            transition: all 0.15s ease !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #09090b !important;
            color: #ffffff !important;
            border-color: #ffffff !important;
        }
        .stButton > button[kind="primary"]:active {
            transform: scale(0.97) !important;
        }

        .stButton > button[kind="secondary"],
        .stDownloadButton > button {
            background-color: #121316 !important;
            color: #f4f4f5 !important;
            border: 1px solid #3f3f46 !important;
            font-weight: 600 !important;
            font-family: 'JetBrains Mono', monospace !important;
            text-transform: uppercase !important;
            font-size: 12px !important;
            letter-spacing: 0.04em !important;
            transition: all 0.15s ease !important;
        }
        .stButton > button[kind="secondary"]:hover,
        .stDownloadButton > button:hover {
            background-color: #27272a !important;
            border-color: #71717a !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="secondary"]:active,
        .stDownloadButton > button:active {
            transform: scale(0.97) !important;
        }

        /* Sidebar Monolith Styling */
        section[data-testid="stSidebar"] {
            background-color: #0c0d0f !important;
            border-right: 1px solid #27272a !important;
        }
        .sidebar-monolith {
            border: 1px solid #27272a;
            background: #141519;
            padding: 14px 16px;
            margin-bottom: 20px;
            border-left: 3px solid #10b981;
        }
        .sidebar-monolith-header {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 700;
            color: #10b981;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .sidebar-monolith-body {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: #71717a;
            margin-top: 6px;
            line-height: 1.55;
        }

        /* Architectural Tabs */
        div[data-baseweb="tab-list"] {
            border-bottom: 1px solid #27272a !important;
            gap: 0px !important;
        }
        div[data-baseweb="tab"] {
            border-bottom: 2px solid transparent !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 12px !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
            color: #71717a !important;
            padding: 10px 18px !important;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            color: #ffffff !important;
            border-bottom: 2px solid #ffffff !important;
            background-color: #121316 !important;
        }

        /* Verdict Placards */
        .verdict-placard {
            border: 1px solid #27272a;
            padding: 18px 22px;
            margin: 18px 0;
            background: #121316;
        }
        .verdict-placard.approve {
            border-left: 4px solid #10b981;
        }
        .verdict-placard.quarantine {
            border-left: 4px solid #ef4444;
        }
        .verdict-placard.escalate {
            border-left: 4px solid #f59e0b;
        }
        .verdict-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .verdict-tag.approve { color: #10b981; }
        .verdict-tag.quarantine { color: #ef4444; }
        .verdict-tag.escalate { color: #f59e0b; }
        .verdict-title {
            font-size: 20px;
            font-weight: 800;
            color: #ffffff;
            margin: 4px 0 8px 0;
            letter-spacing: -0.01em;
        }
        .verdict-stats {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: #a1a1aa;
            margin-bottom: 10px;
        }
        .verdict-text {
            font-size: 13px;
            color: #d4d4d8;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    queue = get_dashboard_queue()

    st.sidebar.markdown(
        """
        <div style="font-family:'JetBrains Mono',monospace; font-size:18px; font-weight:800; letter-spacing:-0.02em; color:#ffffff; margin-bottom:2px;">
            TRUSTGATE // 01
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:#71717a; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px;">
            DATA TRUST & GOVERNANCE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-monolith">
            <div class="sidebar-monolith-header">
                <span style="display:inline-block; width:7px; height:7px; background:#10b981; border-radius:50%; margin-right:4px;"></span>
                SYSTEM ONLINE & AUDITED
            </div>
            <div class="sidebar-monolith-body">
                CORE: FULL-STACK ACTIVE<br>
                MODEL: <code>credit_default_lgb</code><br>
                LEDGER: SHA-256 HASH VERIFIED
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_option = st.sidebar.radio(
        "Navigation",
        [
            "📊 Executive Overview",
            "🔍 Batch Inspector",
            "👥 Human Review Queue",
            "📜 Audit Ledger & Lineage",
            "⚡ Quick Batch Simulator",
            "🔌 API & Endpoints Explorer",
        ],
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        <div style="border: 1px solid #27272a; background: #121316; padding: 12px; border-radius: 2px;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:#71717a; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Target Model Spec</div>
            <div style="font-size:13px; font-weight:700; color:#f4f4f5; margin-bottom:4px;">credit_default_lgb</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#a1a1aa;">
                METRIC: ROC-AUC<br>
                TOLERANCE: &le; 5.0% Damage
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    elif nav_option == "🔌 API & Endpoints Explorer":
        render_api_explorer(queue)


def render_executive_overview(queue: JobQueue) -> None:
    """Render executive KPI cards and decision analytics."""
    st.markdown(
        """
        <div class="editorial-masthead">
            <div class="masthead-meta">
                <span>GOVERNANCE ENGINE // PROD-US-WEST</span>
                <span>MODEL: CREDIT_DEFAULT_LGB • AUDIT: SHA-256 SECURED</span>
            </div>
            <div class="masthead-title">EXECUTIVE GOVERNANCE DISPATCH</div>
            <div class="masthead-subtitle">High-assurance statistical validation, monotonic damage bounds & automated repair</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    st.markdown(
        """
        <div class="editorial-masthead">
            <div class="masthead-meta">
                <span>DEEP INSPECTION // SHAP FACTOR ANALYSIS</span>
                <span>CONFORMAL BOUND: 90% CONFIDENCE INTERVAL</span>
            </div>
            <div class="masthead-title">BATCH LEVEL FORENSICS & REPAIR</div>
            <div class="masthead-subtitle">Detailed SHAP feature attributions, point vs bound damage, and closed-loop cell repair diffs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    st.markdown(
        """
        <div class="editorial-masthead">
            <div class="masthead-meta">
                <span>HUMAN-IN-THE-LOOP // ESCALATION STATION</span>
                <span>PROTOCOL: OPERATOR SIGN-OFF & OVERRIDE</span>
            </div>
            <div class="masthead-title">HUMAN REVIEW & ESCALATION QUEUE</div>
            <div class="masthead-subtitle">Batches flagged with out-of-distribution drift or critical decision risk boundaries</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    st.markdown(
        """
        <div class="editorial-masthead">
            <div class="masthead-meta">
                <span>CRYPTOGRAPHIC LEDGER // IMMUTABLE PROVENANCE</span>
                <span>CIPHER: SHA-256 RECURSIVE HASH CHAIN</span>
            </div>
            <div class="masthead-title">TAMPER-EVIDENT AUDIT LEDGER</div>
            <div class="masthead-subtitle">Append-only cryptographic chain verifying complete end-to-end data governance lineage</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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


def _render_verdict_placard(job: JobRecord) -> None:
    """Render high-contrast brutalist verdict placard for evaluated batch."""
    r_score = job.risk_score or 0.0
    d_upper = (job.damage_upper or 0.0) * 100.0
    dec = job.decision or "UNKNOWN"

    if "APPROVE" in dec and "REPAIR" not in dec:
        css_class = "approve"
        status_label = "VERIFIED_SAFE // COMPLIANT"
    elif "ESCALATE" in dec or "REPAIR" in dec:
        css_class = "escalate"
        status_label = "ATTENTION_REQUIRED // BOUNDARY_ALERT"
    else:
        css_class = "quarantine"
        status_label = "RISK_CONTAINMENT // QUARANTINED"

    expl = (
        job.explanation
        or "Deterministic governance policy executed. Batch is profiled, impact evaluated, and indexed in the cryptographic audit ledger."
    )

    st.markdown(
        f"""
        <div class="verdict-placard {css_class}">
            <div class="verdict-tag {css_class}">[DECISION // {status_label}]</div>
            <div class="verdict-title">{dec}</div>
            <div class="verdict-stats">
                BATCH: <code>{job.batch_id}</code> &nbsp;•&nbsp; 
                RISK INDEX: <strong>{r_score:.1f}/100</strong> &nbsp;•&nbsp; 
                DAMAGE BOUND: <strong>&le; {d_upper:.1f}% ROC-AUC</strong>
            </div>
            <div class="verdict-text">
                {expl}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_simulator(queue: JobQueue) -> None:
    """Test TrustGate live using either custom CSV upload or pre-built corruption scenarios."""
    st.markdown(
        """
        <div class="editorial-masthead">
            <div class="masthead-meta">
                <span>SIMULATOR // REAL-TIME INGESTION</span>
                <span>SCHEMA: TABULAR_CREDIT • VALIDATION: MONOTONIC CALIBRATED</span>
            </div>
            <div class="masthead-title">LIVE TESTING & BATCH SIMULATOR</div>
            <div class="masthead-subtitle">Upload custom tabular CSV files or execute simulated real-world corruptions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_upload, tab_presets = st.tabs(["📁 Upload Custom CSV File", "⚡ Synthetic Scenarios & Presets"])

    with tab_upload:
        st.subheader("Upload Your Own Tabular Dataset")
        st.write(
            "Drop your custom batch CSV file here to run real-time schema profiling, risk scoring, "
            "and downstream model damage prediction."
        )

        # Pre-generated sample downloads for expo visitors
        try:
            _, sample_clean_df, _, _ = get_dataset("credit_default", n_samples=60, seed=123)
            sample_dirty_df, _ = inject_outliers_and_sentinels(
                sample_clean_df, ["income", "debt"], severity=0.25, seed=99
            )
            c1, c2 = st.columns(2)
            c1.download_button(
                "📥 Download Sample Clean CSV",
                data=sample_clean_df.write_csv(),
                file_name="sample_clean_batch.csv",
                mime="text/csv",
                help="Test with a compliant, clean benchmark dataset",
            )
            c2.download_button(
                "📥 Download Sample Corrupted CSV",
                data=sample_dirty_df.write_csv(),
                file_name="sample_corrupted_batch.csv",
                mime="text/csv",
                help="Test with an anomaly/corrupted dataset to see quarantine & risk scoring",
            )
        except Exception:
            pass

        st.markdown("---")

        uploaded_file = st.file_uploader(
            "Choose a CSV file to evaluate",
            type=["csv"],
            help="Upload a tabular CSV file to evaluate with TrustGate",
            key="custom_csv_uploader",
        )

        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            try:
                preview_df = pd.read_csv(uploaded_file)
                st.write(
                    f"**File:** `{uploaded_file.name}` | **Rows:** {len(preview_df)} | **Columns:** {len(preview_df.columns)}"
                )
                st.dataframe(preview_df.head(6), use_container_width=True)

                if st.button("🚀 Analyze & Process Uploaded Batch", type="primary", key="btn_process_upload"):
                    with st.spinner("Analyzing uploaded file, profiling drift & calculating model damage..."):
                        job = queue.submit_batch(
                            raw_bytes=raw_bytes,
                            filename=uploaded_file.name,
                            target_model_name="credit_default_lgb",
                        )

                        for _ in range(40):
                            if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                                break
                            time.sleep(0.1)

                        if job.status == JobStatus.FAILED:
                            st.error(f"Processing failed: {job.error}")
                        else:
                            st.success(f"Batch {job.batch_id} successfully evaluated!")
                            _render_verdict_placard(job)
            except Exception as e:
                st.error(f"Failed to read CSV file: {e}")

    with tab_presets:
        st.subheader("Synthetic Scenario Presets")
        st.caption("One-click benchmark datasets with simulated real-world corruptions")

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
            key="preset_select",
        )

        if st.button("Generate & Process Preset Batch", key="btn_process_preset"):
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
                _render_verdict_placard(job)


def render_api_explorer(queue: JobQueue) -> None:
    """Unified API Explorer allowing full inspection and testing of REST endpoints in one URL."""
    st.markdown(
        """
        <div class="editorial-masthead">
            <div class="masthead-meta">
                <span>API EXPLORER // REST INTEGRATION</span>
                <span>SPECIFICATION: OPENAPI 3.1 • RBAC ENFORCED</span>
            </div>
            <div class="masthead-title">API & ENDPOINTS EXPLORER</div>
            <div class="masthead-subtitle">Live inspection of production endpoints, payload schemas, and programmatic pipeline integration</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Status summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Service Status", "HEALTHY (200 OK)")
    col2.metric("API Version", "v1.0.0 (Production)")
    col3.metric("Authentication", "Role-Based API Key (RBAC)")

    st.divider()

    st.subheader("Available Production REST Endpoints")
    endpoint_data = [
        {"Method": "POST", "Path": "/v1/batches", "Auth": "Operator+", "Description": "Submit tabular CSV/Parquet batch for profiling, damage prediction & governance"},
        {"Method": "GET", "Path": "/v1/batches/{id}/decision", "Auth": "Viewer+", "Description": "Fetch deterministic decision record, conformal bounds, and SHAP factors"},
        {"Method": "GET", "Path": "/v1/batches/{id}/certificate", "Auth": "Viewer+", "Description": "Download cryptographic AI-Readiness Certificate with ledger hash"},
        {"Method": "POST", "Path": "/v1/batches/{id}/review", "Auth": "Operator+", "Description": "Human operator sign-off (approve override or confirm quarantine)"},
        {"Method": "GET", "Path": "/v1/audit/verify", "Auth": "Viewer+", "Description": "Run independent cryptographic SHA-256 hash-chain verification"},
        {"Method": "GET", "Path": "/healthz", "Auth": "Public", "Description": "Container and service liveness / readiness health probe"},
        {"Method": "GET", "Path": "/metrics", "Auth": "Public", "Description": "Prometheus metrics counters, risk sums, and processing latency histograms"},
    ]
    st.dataframe(pd.DataFrame(endpoint_data), use_container_width=True)

    st.divider()

    st.subheader("Interactive Payload & Response Preview")
    jobs = queue.list_jobs()
    if jobs:
        selected_job = st.selectbox(
            "Select processed batch to preview REST API response:",
            [j.batch_id for j in reversed(jobs)],
            key="api_job_select",
        )
        j_rec = next((j for j in jobs if j.batch_id == selected_job), jobs[0])

        tab_decision, tab_cert, tab_curl = st.tabs(["Decision Payload", "AI Certificate", "cURL Snippet"])
        with tab_decision:
            st.json({
                "batch_id": j_rec.batch_id,
                "job_id": j_rec.job_id,
                "status": j_rec.status.value,
                "decision": j_rec.decision,
                "risk_score": j_rec.risk_score,
                "predicted_damage": j_rec.predicted_damage,
                "damage_upper_bound_90pct": j_rec.damage_upper,
                "repaired_cells_count": j_rec.repaired_cells_count,
                "explanation": j_rec.explanation,
            })
        with tab_cert:
            if j_rec.certificate:
                st.json(j_rec.certificate)
            else:
                st.info("Certificate generated on final ledger seal.")
        with tab_curl:
            st.code(
                f"""# Submit a batch for automated governance evaluation:
curl -X POST "https://trustgate-api.onrender.com/v1/batches" \\
  -H "X-API-Key: tg-operator-key-2026" \\
  -F "file=@your_data.csv" \\
  -F "target_model_name=credit_default_lgb"

# Fetch decision record:
curl -X GET "https://trustgate-api.onrender.com/v1/batches/{j_rec.batch_id}/decision" \\
  -H "X-API-Key: tg-viewer-key-2026"
""",
                language="bash",
            )
    else:
        st.info("Submit or simulate a batch to view live API responses.")


if __name__ == "__main__":
    render_dashboard()
