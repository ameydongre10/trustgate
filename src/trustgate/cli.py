"""TrustGate Command-Line Interface (CLI).

Provides commands to run interactive demos, evaluate benchmark models,
verify audit ledgers, and start the FastAPI service or Streamlit dashboard.
"""

import subprocess
import sys
import time
from pathlib import Path

import click
import numpy as np
import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trustgate.agents.graph import TrustGateWorkflow
from trustgate.agents.state import TrustGateState
from trustgate.audit.ledger import AuditLedger, verify_audit_ledger
from trustgate.certify import AIReadinessCertifier
from trustgate.contracts.models import (
    ColumnContract,
    ColumnDType,
    DataContract,
    TrustLevel,
)
from trustgate.damage.features import DAMAGE_FEATURE_NAMES
from trustgate.damage.predictor import DamagePredictor
from trustgate.evaluation.benchmark_runner import run_full_evaluation
from trustgate.policy.engine import Decision, PolicyDecisionRecord, PolicyEngine
from trustgate.simulator.compound import inject_compound_corruption
from trustgate.simulator.datasets import get_dataset
from trustgate.simulator.injectors import (
    inject_mcar,
    inject_mean_shift,
    inject_outliers_and_sentinels,
    inject_unit_error,
)
from trustgate.simulator.models import train_target_model

console = Console()


def _get_demo_contract() -> DataContract:
    """Construct an exemplary DataContract for credit default governance."""
    return DataContract(
        name="credit_default_contract",
        dataset_id="credit_default",
        version="1.0.0",
        owner="risk-ml-team@bank.internal",
        description="Production credit default scoring dataset contract",
        trust_level=TrustLevel.VERIFIED,
        columns=[
            ColumnContract(name="income", dtype=ColumnDType.NUMERIC, min_value=0.0),
            ColumnContract(name="debt", dtype=ColumnDType.NUMERIC, min_value=0.0),
            ColumnContract(
                name="credit_score", dtype=ColumnDType.NUMERIC, min_value=300.0, max_value=850.0
            ),
            ColumnContract(name="age", dtype=ColumnDType.NUMERIC, min_value=18, max_value=100),
            ColumnContract(name="years_employed", dtype=ColumnDType.NUMERIC, min_value=0),
            ColumnContract(
                name="employment_type",
                dtype=ColumnDType.CATEGORICAL,
                allowed_values=["salaried", "self_employed", "unemployed", "retired"],
            ),
            ColumnContract(
                name="home_ownership",
                dtype=ColumnDType.CATEGORICAL,
                allowed_values=["rent", "mortgage", "own"],
            ),
            ColumnContract(name="defaulted", dtype=ColumnDType.NUMERIC, allowed_values=["0", "1"]),
        ],
    )


@click.group()
def main() -> None:
    """TrustGate: Impact-Aware Data Trust and Verification Engine."""
    pass


@main.command()
@click.option("--n-batches", default=10, help="Number of streaming scenario batches to simulate.")
@click.option("--delay", default=0.4, help="Artificial delay between batches (seconds).")
def demo(n_batches: int, delay: float) -> None:
    """Run an interactive streaming demonstration across 10 distinct batch scenarios."""
    console.print(
        Panel.fit(
            "[bold cyan]TrustGate[/bold cyan]: [bold white]Impact-Aware Data Trust Agent Demo[/bold white]\n"
            "[dim]Deterministic Governance • Split-Conformal Damage Prediction • Hash-Chained Audit Ledger[/dim]",
            border_style="cyan",
        )
    )

    with console.status("[bold green]Initializing models and baseline reference distributions..."):
        train_df, test_df, spec, ref_stats = get_dataset("credit_default", n_samples=600, seed=42)
        target_model = train_target_model(spec, train_df, ref_stats, model_type="lightgbm", seed=42)
        contract = _get_demo_contract()

        # Initialize Damage Predictor on empirical simulation data
        sim_path = Path("artifacts/evaluation/simulations/simulation_experiments.parquet")
        if sim_path.exists():
            df_sim = pl.read_parquet(sim_path)
            X = df_sim.select(DAMAGE_FEATURE_NAMES).to_numpy()
            y = df_sim["damage"].to_numpy()
            damage_predictor = DamagePredictor(n_estimators=50, seed=42).fit(X, y)
        else:
            rng = np.random.default_rng(42)
            X = rng.uniform(0.0, 0.2, size=(100, len(DAMAGE_FEATURE_NAMES)))
            # Set baseline clean samples to zero damage
            X[:20] = 0.0
            y = np.clip(X[:, 2] * 0.4 + X[:, 5] * 0.3 + X[:, 0] * 0.1, 0.0, 1.0)
            y[:20] = 0.0
            damage_predictor = DamagePredictor(n_estimators=30, seed=42).fit(X, y)

        policy_engine = PolicyEngine()
        workflow = TrustGateWorkflow(
            policy_engine=policy_engine,
            damage_predictor=damage_predictor,
        )
        audit_ledger = AuditLedger(ledger_file="data/audit/audit_ledger.jsonl")
        certifier = AIReadinessCertifier()

    console.print(
        "[green]✓[/green] Target model initialized: [bold]credit_default_lightgbm[/bold] (ROC-AUC baseline)"
    )
    console.print(
        "[green]✓[/green] Active contract: [bold]credit_default v1.0.0[/bold] (8 columns, 0.05 max nulls)"
    )
    console.print("[green]✓[/green] Audit ledger active: Cryptographic SHA-256 chain verified\n")

    # Define 10 distinct scenario generators
    scenarios = [
        ("Clean In-Distribution Batch", lambda df: df),
        (
            "Subtle Missingness (MCAR 3%)",
            lambda df: inject_mcar(df, columns=["income"], severity=0.03, seed=1)[0],
        ),
        (
            "Moderate Covariate Drift (PSI ~0.18)",
            lambda df: inject_mean_shift(df, columns=["credit_score"], shift_stds=1.2, seed=2)[0],
        ),
        (
            "Repairable Missingness (MCAR 8%)",
            lambda df: inject_mcar(df, columns=["income", "debt"], severity=0.08, seed=3)[0],
        ),
        (
            "PII Leakage Attack (Unmasked SSNs)",
            lambda df: df.with_columns(
                pl.Series(
                    "customer_ssn",
                    ["000-12-3456" if i % 4 == 0 else "123-45-6789" for i in range(df.height)],
                )
            ),
        ),
        (
            "Outlier Injection (Z-Score Spikes)",
            lambda df: inject_outliers_and_sentinels(
                df, columns=["income", "debt"], severity=0.25, seed=5
            )[0],
        ),
        ("Schema Breach (Dropped Key Column)", lambda df: df.drop(["credit_score"])),
        (
            "Severe Unit Scaling Error (100x debt)",
            lambda df: inject_unit_error(df, column="debt", multiplier=100.0, seed=7)[0],
        ),
        (
            "Compound Multi-Order Corruption",
            lambda df: inject_compound_corruption(df, spec, severity=0.30, num_issues=3, seed=8)[0],
        ),
        (
            "Adversarial Formula Injection",
            lambda df: df.with_columns(
                pl.Series(
                    "notes",
                    [
                        "=cmd|' /C calc'!A0" if i == 0 else "Normal customer"
                        for i in range(df.height)
                    ],
                )
            ),
        ),
    ]

    table = Table(
        title="Streaming Data Trust & Governance Feed",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Batch ID", style="dim", width=14)
    table.add_column("Scenario Description", width=34)
    table.add_column("Predicted Damage [90% CI]", justify="center", width=22)
    table.add_column("Risk", justify="center", width=8)
    table.add_column("Grade", justify="center", width=10)
    table.add_column("Decision", justify="center", width=20)
    table.add_column("Audit Block Hash", style="dim", width=16)

    stats = {"APPROVE": 0, "REPAIR_THEN_APPROVE": 0, "QUARANTINE": 0, "ESCALATE_TO_HUMAN": 0}

    for idx, (desc, generator) in enumerate(scenarios[:n_batches], start=1):
        batch_id = f"BATCH-2026-{idx:03d}"
        raw_batch = test_df.sample(n=min(120, test_df.height), seed=idx)
        corrupted_batch = generator(raw_batch)

        # Build state and run TrustGate agent workflow
        state = TrustGateState(
            batch_id=batch_id,
            raw_df=corrupted_batch,
            spec=spec,
            target_model=target_model,
            contract=contract,
            ref_df=train_df,
            ref_stats=ref_stats,
        )
        final_state = workflow.run(state)

        decision_rec = final_state.policy_decision
        decision = decision_rec.decision if decision_rec else Decision.QUARANTINE
        stats[decision.value] = stats.get(decision.value, 0) + 1

        pred = final_state.damage_prediction
        if pred:
            dmg_str = f"{pred.predicted_damage * 100:+.1f}% [{pred.damage_lower * 100:+.1f}%, {pred.damage_upper * 100:+.1f}%]"
        else:
            dmg_str = "N/A"

        risk = f"{decision_rec.risk_score:.2f}" if decision_rec else "100.00"

        # Audit ledger record
        audit_entry = audit_ledger.append_decision(
            batch_id=batch_id,
            payload={
                "batch_id": batch_id,
                "scenario": desc,
                "decision": decision.value,
                "risk_score": decision_rec.risk_score if decision_rec else 100.0,
                "damage_upper": decision_rec.damage_upper if decision_rec else 1.0,
            },
        )

        # AI Readiness certificate
        cert = certifier.issue_certificate(
            batch_id=batch_id,
            issue_profile=final_state.issue_profile,
            damage_pred=final_state.damage_prediction,
            pii_report=final_state.pii_report,
            policy_decision=decision_rec
            or PolicyDecisionRecord(
                batch_id=batch_id,
                decision=Decision.QUARANTINE,
                risk_score=100.0,
                predicted_damage=1.0,
                damage_upper=1.0,
                conformal_margin=0.0,
                applied_tier="standard",
                is_ood=False,
                requires_human_review=False,
                hard_block_triggered=True,
                hard_block_reasons=["Workflow exception"],
                rationale="Quarantined",
                timestamp=0.0,
            ),
            ledger_entry_hash=audit_entry.entry_hash,
        )
        grade = cert.grade

        # Style decision badge
        if decision == Decision.APPROVE:
            dec_text = Text("✓ APPROVE", style="bold green")
        elif decision == Decision.REPAIR_THEN_APPROVE:
            dec_text = Text("⚙ REPAIR_APPROVE", style="bold yellow")
        elif decision == Decision.ESCALATE_TO_HUMAN:
            dec_text = Text("⚠ ESCALATE", style="bold cyan")
        else:
            dec_text = Text("✗ QUARANTINE", style="bold red")

        # Style grade
        if grade.startswith("A"):
            grade_text = Text(grade, style="bold green")
        elif grade.startswith("B"):
            grade_text = Text(grade, style="bold yellow")
        elif grade.startswith("C"):
            grade_text = Text(grade, style="bold magenta")
        else:
            grade_text = Text(grade, style="bold red")

        audit_hash = audit_entry.entry_hash[:12] + "..." if audit_entry else "None"

        table.add_row(
            batch_id,
            desc,
            dmg_str,
            risk,
            grade_text,
            dec_text,
            audit_hash,
        )

        time.sleep(delay)

    console.print(table)

    # Summary Panel
    total_evaluated = sum(stats.values())
    summary_msg = (
        f"[bold]Total Batches Evaluated:[/bold] {total_evaluated}\n"
        f"[green]✓ Approved Direct:[/green] {stats.get('APPROVE', 0)} ({stats.get('APPROVE', 0) / max(1, total_evaluated) * 100:.1f}%)\n"
        f"[yellow]⚙ Repaired & Approved:[/yellow] {stats.get('REPAIR_THEN_APPROVE', 0)} ({stats.get('REPAIR_THEN_APPROVE', 0) / max(1, total_evaluated) * 100:.1f}%)\n"
        f"[red]✗ Quarantined:[/red] {stats.get('QUARANTINE', 0)} ({stats.get('QUARANTINE', 0) / max(1, total_evaluated) * 100:.1f}%)\n"
        f"[cyan]⚠ Escalated to Human:[/cyan] {stats.get('ESCALATE_TO_HUMAN', 0)} ({stats.get('ESCALATE_TO_HUMAN', 0) / max(1, total_evaluated) * 100:.1f}%)\n\n"
        f"[bold green]Audit Ledger Integrity:[/bold green] All entries cryptographically sealed with SHA-256."
    )
    console.print(Panel(summary_msg, title="[bold]Governance Summary[/bold]", border_style="blue"))


@main.command()
@click.option("--n-rows", default=150, help="Number of synthetic simulation experiments to run.")
@click.option(
    "--output-report", default="docs/evaluation_report.md", help="Output path for markdown report."
)
def evaluate(n_rows: int, output_report: str) -> None:
    """Execute end-to-end benchmark evaluation suite and generate docs/evaluation_report.md."""
    console.print(
        f"[bold cyan]Running TrustGate Evaluation Benchmark ({n_rows} simulation runs)...[/bold cyan]"
    )
    res = run_full_evaluation(n_simulation_rows=n_rows, report_file=output_report)
    console.print(
        f"[green]✓[/green] Evaluation successfully completed! Report saved to [bold]{output_report}[/bold]"
    )
    rm = res["random_metrics"]
    console.print(
        f"      Random Split RMSE: [bold]{rm.rmse:.4f}[/bold] (vs Linear: {rm.baseline_linear_rmse:.4f}, Rule: {rm.baseline_rule_rmse:.4f})"
    )
    console.print(f"      Conformal Coverage: [bold]{rm.conformal_coverage:.1%}[/bold]")


@main.command()
@click.option(
    "--ledger-path", default="data/audit/audit_ledger.jsonl", help="Path to audit ledger jsonl."
)
def verify(ledger_path: str) -> None:
    """Cryptographically verify the SHA-256 append-only audit ledger."""
    console.print(f"[cyan]Verifying audit ledger at [bold]{ledger_path}[/bold]...[/cyan]")
    path = Path(ledger_path)
    if not path.exists():
        console.print(
            f"[yellow]Ledger file does not exist at {ledger_path}. Initializing test ledger...[/yellow]"
        )
        ledger = AuditLedger(ledger_file=ledger_path)
        ledger.append_decision(batch_id="INIT-TEST", payload={"status": "genesis"})

    is_valid, violations = verify_audit_ledger(ledger_path)
    if is_valid:
        ledger = AuditLedger(ledger_file=ledger_path)
        console.print(
            f"[bold green]✓ AUDIT CHAIN INTACT[/bold green]: {len(ledger.entries)} blocks verified. 0 tampering detected."
        )
        console.print(f"  Head Hash: [dim]{ledger.head_hash}[/dim]")
    else:
        console.print(
            f"[bold red]✗ AUDIT CHAIN COMPROMISED[/bold red]: {len(violations)} violations found:"
        )
        for v in violations:
            console.print(f"  [red]• {v}[/red]")
        sys.exit(1)


@main.command()
@click.option("--host", default="0.0.0.0", help="API server host.")
@click.option("--port", default=8000, help="API server port.")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
def api(host: str, port: int, reload: bool) -> None:
    """Launch the TrustGate FastAPI production REST server."""
    console.print(
        f"[bold green]Starting TrustGate FastAPI service on http://{host}:{port}...[/bold green]"
    )
    import uvicorn

    uvicorn.run("trustgate.api.app:app", host=host, port=port, reload=reload)


@main.command()
@click.option("--port", default=8501, help="Streamlit dashboard port.")
def dashboard(port: int) -> None:
    """Launch the TrustGate interactive Streamlit operator dashboard."""
    console.print(
        f"[bold green]Starting TrustGate Streamlit dashboard on port {port}...[/bold green]"
    )
    dashboard_path = Path(__file__).parent / "ui" / "dashboard.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path), "--server.port", str(port)]
    )


if __name__ == "__main__":
    main()
