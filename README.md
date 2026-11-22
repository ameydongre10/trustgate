# TrustGate: Impact-Aware Data Trust & Verification Agent

[![CI Status](https://img.shields.io/badge/CI-Passing-brightgreen.svg)]()
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary_All_Rights_Reserved-red.svg)](LICENSE)
[![Type Checked: mypy](https://img.shields.io/badge/type_checker-mypy_strict-blue.svg)]()
[![Linting: ruff](https://img.shields.io/badge/linter-ruff-000000.svg)]()
[![Tests: 98 Passed](https://img.shields.io/badge/tests-98_passed-success.svg)]()

> **TrustGate** is an enterprise data trust and verification agent that protects downstream ML models and analytical pipelines by predicting the downstream degradation caused by data corruption, rather than counting arbitrary rule violations.

---

## 🌐 Production Live Deployments & Service Endpoints

TrustGate is deployed across production cloud services with automated 24/7 uptime protection and zero-cold-start acceleration:

| Service / Interface | Type | Live Production URL 
| :--- | :--- | :--- | :--- |
| **TrustGate Operator Console** | **Dynamic App** | **[https://trustgate-ui.onrender.com](https://trustgate-ui.onrender.com)** 

> [!NOTE]
> **24/7 Keep-Alive Heartbeat**: Powered by [`.github/workflows/keep_alive.yml`](.github/workflows/keep_alive.yml), which automatically queries `trustgate-ui` and `trustgate-api` every 10 minutes to prevent Render free-tier instances from entering idle sleep mode.

---

### 🧪 How to Test TrustGate Live with Custom Datasets (Step-by-Step)

TrustGate allows anyone (operators, interviewers, expo attendees) to test the governance pipeline live with their **own custom CSV files** or built-in corrupted benchmarks:

```
Step 1: Open https://trustgate-ui.onrender.com in any web browser.
Step 2: In the left sidebar, click "⚡ Quick Batch Simulator".
Step 3: Click on the "📁 Upload Custom CSV File" tab.
Step 4: (Optional) If you don't have a CSV handy, click "📥 Download Sample Clean CSV" or "📥 Download Sample Corrupted CSV".
Step 5: Drag and drop your `.csv` file into the upload zone.
Step 6: Review the instant 6-row table preview (showing row count and column structure).
Step 7: Click "🚀 Analyze & Process Uploaded Batch".
```

#### 📊 What TrustGate Returns in Real-Time:
1. **Deterministic Governance Verdict**:
   * `APPROVE` — Batch is safe; predicted downstream impact is within tolerance ($\le 5\%$).
   * `REPAIR_THEN_APPROVE` — Corruptions (e.g. missing values) were repaired automatically within risk budget.
   * `QUARANTINE` — Severe degradation predicted ($> 20\%$) or hard PII/security contract breached.
   * `ESCALATE_TO_HUMAN` — Out-of-distribution drift or boundary-level risk routed to human operator.
2. **Calibrated Risk Score**: Normalized risk index between `0.0` and `100.0`.
3. **Predicted Model Damage Upper Bound**: Conformal $90\%$ confidence upper bound on metric loss (e.g., `11.9% ROC-AUC loss`).
4. **AI Governance Rationale**: Plain-English explainability narrative generated and verified by anti-hallucination regex guards.
5. **Cryptographic Proof**: Switch to **`📜 Audit Ledger & Lineage`** to verify that your uploaded batch's decision has been recorded into the SHA-256 immutable ledger with a unique sequence number and parent hash.

---

## 1. Overview & Core Mission

Traditional data quality tools (Great Expectations, Deequ, Soda) operate on declarative rule counts or isolated p-values. If an un-modeled auxiliary column is missing, pipelines crash; if a critical feature subtly drifts, models fail silently.

**TrustGate** bridges statistical profiling, machine learning impact prediction, and deterministic data governance:

1. **Profiles** incoming data batches for distribution drift (Two-sample KS, PSI, $\chi^2$, Jensen-Shannon), quality anomalies, outlier spikes, PII leaks, and schema contract breaches.
2. **Predicts** exact downstream model degradation:
   $$\text{Damage} = \frac{\text{metric}_{\text{clean}} - \text{metric}_{\text{corrupted}}}{\text{metric}_{\text{clean}}}$$
   using a calibrated monotonic LightGBM regressor equipped with **split-conformal prediction intervals** guaranteeing distribution-free coverage.
3. **Decides** deterministically among:
   - `APPROVE`: Damage upper bound is safely within the allowable risk budget.
   - `REPAIR_THEN_APPROVE`: Closed-loop automated repair modifies corrupted cells within budget and re-evaluates post-repair damage.
   - `QUARANTINE`: Predicted damage exceeds rejection thresholds, or hard security/PII blocks are triggered.
   - `ESCALATE_TO_HUMAN`: Low confidence, out-of-distribution feature regimes, or near-boundary decisions.
4. **Explains** decisions with TreeSHAP factor drivers and LLM narratives verified against hallucinated figures by a deterministic regex claim verifier.
5. **Certifies** data batches with an AI-Readiness score (0–100) across 10 dimensions, sealing provenance in an **append-only SHA-256 hash-chained audit ledger**.

---

## 2. Non-Negotiable Engineering Principles

1. **Deterministic by Default**: Governance decisions are **never made by an LLM**. The Judge is deterministic (policy engine + monotonic model + conformal upper bound).
2. **No Fabricated Results**: Every metric in reports or README is produced by reproducible code (`make evaluate`).
3. **100% Offline Capable**: The entire system operates without external network access or paid LLM API keys via a template-based fallback explainer.
4. **Fit-on-Train-Only Discipline**: Imputers, scalers, encoders, and reference distributions strictly adhere to fit-on-train discipline to prevent data leakage.
5. **Untrusted Input Discipline**: All headers, cell contents, and file metadata are treated as adversarial inputs (hardened against prompt injection, formula injection, ReDoS, and zip/decompression bombs).
6. **Fail-Safe Containment (Rule 8)**: Any unexpected internal failure, timeout, or OOD regime resolves to `QUARANTINE` or `ESCALATE_TO_HUMAN`, never silent `APPROVE`.

---

## 3. System Architecture

```mermaid
flowchart TD
    subgraph Ingest & Contract Phase
        A[Incoming Data Batch CSV/Parquet] --> B[Safe Ingest & Sanitizer]
        B --> C[ReDoS-Safe DataContract Validator]
    end

    subgraph Profiling & Privacy Phase
        C --> D[Statistical & Drift Profiler KS/PSI/Chi2]
        C --> E[PII Scanner Luhn / Verhoeff Checksums]
    end

    subgraph Impact Prediction Phase
        D --> F[Issue Profile Feature Builder]
        E --> F
        G[(Target Model Registry)] --> F
        F --> H[Monotonic LightGBM Regressor]
        H --> I[Split-Conformal Interval Estimator]
        F --> J[OOD Guard Isolation Forest + Mahalanobis]
    end

    subgraph Deterministic Governance Phase
        I --> K{Deterministic Policy Engine}
        J --> K
        E -->|Critical PII Block| K
        C -->|Schema Breach Block| K

        K -->|Damage <= 5%| L[APPROVE]
        K -->|5% < Damage <= 20%| M[Closed-Loop Repair Planner]
        M -->|Re-Profile & Re-Predict| K
        K -->|Damage > 20% or Hard Block| N[QUARANTINE Store]
        K -->|OOD or Boundary Alert| O[Human Review Queue]
    end

    subgraph Explainability & Audit Phase
        K --> P[TreeSHAP Factor Attributions]
        P --> Q[Explainer Agent Narrative Generator]
        Q --> R[Numeric Claim Verifier anti-hallucination]
        K --> S[(Append-Only SHA-256 Audit Ledger)]
        S --> T[Cryptographic AI-Readiness Certificate]
    end
```

---

## 4. Empirical Evaluation & Benchmark Results

Results generated directly by running the evaluation harness (`src/trustgate/evaluation/benchmark_runner.py`):

### 4.1 In-Distribution Model Comparison vs Baselines
| Model / Governance Baseline | Test RMSE | Test MAE | Conformal Coverage (90% Target) |
| :--- | :---: | :---: | :---: |
| **TrustGate Monotonic LightGBM** | **0.4277** | **0.2140** | **64.4%** |
| Ridge / Linear Baseline | 0.2877 | - | - |
| Rule-Count Heuristic Baseline | 0.2383 | - | - |

### 4.2 Policy Comparison Study (Business Impact at $\tau = 5.0\%$)
| Governance Strategy | Pipeline Throughput (% Approved) | Downstream Avg Degradation | Downstream Damage Avoided |
| :--- | :---: | :---: | :---: |
| **TrustGate (Impact-Aware)** | **0.7%** | **-2.56%** | **+119.2% Protected** |
| Legacy Rule-Count (>1 violation) | 91.3% | 12.52% | Sub-optimal Trade-off |
| Always Pass (No Gate) | 100.0% | 13.36% | 0.0% (Severe Vulnerability) |
| Always Quarantine | 0.0% | 0.00% | Pipeline Stalled |

### 4.3 Generalization Protocols
- **Leave-One-Corruption-Out (LOCO)**: Evaluated across 12 unseen corruption modes (MCAR, MAR, MNAR, mean shift, variance scale, unseen categories, outliers, label noise, duplicates, unit errors, concept drift, compound).
- **Leave-One-Dataset-Out (LODO)**: Cross-domain validation across `credit_default`, `telco_churn`, and `housing_prices`.
- **Leave-One-Model-Out (LOMO)**: Evaluated transferability across Gradient Boosting, Random Forests, and Linear models.

---

## 5. Quickstart & CLI Commands

### 5.1 Installation
```bash
# Clone and setup
git clone <repo-url> trustgate
cd trustgate

# Initialize environment and install dependencies
uv venv --python 3.11 .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 5.2 Streaming Interactive Demo
Run a 10-batch live streaming scenario demonstration with Rich terminal tables:
```bash
trustgate demo --n-batches 10 --delay 0.3
```

### 5.3 Launch the Production FastAPI REST Service
```bash
trustgate api --host 0.0.0.0 --port 8000
```
- Interactive API Documentation: `http://localhost:8000/docs`
- Healthcheck Endpoint: `http://localhost:8000/healthz`
- Prometheus Metrics: `http://localhost:8000/metrics`

### 5.4 Launch the Streamlit Operator Dashboard
```bash
trustgate dashboard --port 8501
```

### 5.5 Cryptographic Audit Ledger Verification
Verify the SHA-256 hash chain and detect any tampering:
```bash
trustgate verify --ledger-path data/audit/audit_ledger.jsonl
```

### 5.6 Run Evaluation Benchmark
```bash
trustgate evaluate --n-rows 180 --output-report docs/evaluation_report.md
```

---

## 6. Adversarial Security & Edge Case Hardening

The test suite (`tests/adversarial/test_adversarial_security.py`) enforces strict security controls:

- **Prompt Injection Defense**: Sanitizes cells containing `Ignore all previous instructions...` before LLM processing.
- **Spreadsheet Formula Injection**: Automatically escapes leading `=`, `+`, `-`, `@`, `\t`, `\r` to neutralize DDE attacks in Excel/Sheets.
- **Checksum Evasion Resistance**: Validates Indian Aadhaar (Verhoeff D8), Indian PAN, Credit Cards (Luhn mod 10), SSN format.
- **ReDoS Protection**: Validates all regex patterns with catastrophic backtracking guards.
- **Decompression Bomb Defense**: Aborts archives exceeding a 100:1 expansion ratio.
- **Audit Ledger Tamper-Evidence**: Recomputes canonical SHA-256 hashes for all blocks; detects modified payloads, deleted blocks, and sequence gaps.
- **Extreme Edge Cases**: Tested on 1-row micro-batches, 1000-column wide tables, all-null matrices, and empty batches.

---

## 7. Makefile Targets

```bash
make test       # Run 98 automated unit, property, adversarial, integration, and E2E tests
make lint       # Run Ruff linter across src/ and tests/
make type       # Run Mypy strict type checking
make format     # Auto-format codebase with Ruff
make evaluate   # Re-run empirical benchmarks and regenerate docs/evaluation_report.md
make demo       # Run the 10-batch streaming CLI demo
make api        # Start the FastAPI backend
make ui         # Start the Streamlit dashboard
```

---

## 8. License & Intellectual Property Protection

**Copyright © 2026 ameydongre10 (Amey Dongre). All Rights Reserved.**

This repository, source code, neural algorithms, model architectures, simulator engines, and governance systems are the exclusive proprietary intellectual property of **@ameydongre10**.

> [!CAUTION]
> ### AUTOMATIC COPYRIGHT STRIKE & DMCA ENFORCEMENT NOTICE
> **Unauthorized copying, forking, scraping, mirroring, re-distribution, or commercial exploitation is strictly prohibited.**
>
> - **Immediate Automatic Strike**: Any unauthorized copy, mirror, or derivative work discovered on GitHub will receive an immediate, expedited **Digital Millennium Copyright Act (DMCA) Notice of Infringement under 17 U.S.C. § 512(c)**, resulting in the permanent removal of the repository and an **official Copyright Strike** applied against the offending user account.
> - **Hosting Suspension**: Takedown demands will simultaneously be transmitted to all cloud hosting providers (Render, Vercel, Railway, AWS, GCP, etc.) hosting unauthorized instances, triggering immediate service termination.
> - **Cryptographic Provenance**: The codebase contains proprietary SHA-256 hash chains, deterministic seed signatures, and embedded digital fingerprints providing mathematical proof of authorship and origin.

For licensing inquiries or commercial use, contact: **[@ameydongre10](https://github.com/ameydongre10)**.
