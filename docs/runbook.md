# TrustGate Operator Runbook & Operations Guide

This runbook guides operators, MLOps engineers, and data governors in deploying, configuring, monitoring, and troubleshooting **TrustGate**.

---

## 1. Quick Start & Service Execution

### 1.1 Local Installation
```bash
# Clone and enter directory
git clone <repo-url> trustgate
cd trustgate

# Initialize virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .\.venv\Scripts\Activate.ps1

# Install package and dependencies in editable mode
pip install -e ".[dev]"
```

### 1.2 Interactive Streaming Demo
To observe the TrustGate agent stream 10 live scenario batches (clean, drift, PII, schema breaches, compound corruptions) through live split-conformal inference and deterministic policy gates:
```bash
trustgate demo --n-batches 10 --delay 0.3
```

### 1.3 Launching the Production REST API
```bash
trustgate api --host 0.0.0.0 --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Healthcheck Endpoint: `http://localhost:8000/healthz`
- Prometheus Metrics: `http://localhost:8000/metrics`

### 1.4 Launching the Streamlit Operator Dashboard
```bash
trustgate dashboard --port 8501
```
Navigate to `http://localhost:8501` to access:
- **Executive Overview**: Real-time KPI counters, approval rates, damage prevented.
- **Batch Inspector**: Deep inspection of conformal intervals, SHAP factor drivers, and dirty vs repaired cell previews.
- **Human Review Queue**: Interactive operator approval/override interface for escalated batches.
- **Audit Ledger**: Real-time visual timeline of SHA-256 blocks and hash-chain verification.
- **Batch Simulator**: Live scenario injection and synthetic batch testing.

---

## 2. API Authentication & RBAC

The TrustGate REST API implements role-based API key authentication via the `X-API-Key` HTTP header.

| Role | Default Key | Permissions |
| :--- | :--- | :--- |
| **Viewer** | `tg-viewer-key-2026` | Read batches, inspect decisions, download certificates, check audit chain. |
| **Operator** | `tg-operator-key-2026` | All Viewer permissions + Submit batches, perform human review sign-offs, register contracts. |
| **Admin** | `tg-admin-key-2026` | Full access + Modify thresholds, purge data, administrative controls. |

*Production Note*: Keys should be injected via environment variables (`TG_API_KEY_ADMIN`, etc.).

---

## 3. Cryptographic Audit Chain Verification

To independently verify that the append-only audit ledger has not been tampered with or modified:
```bash
trustgate verify --ledger-path data/audit/audit_ledger.jsonl
```
Or via HTTP:
```bash
curl -H "X-API-Key: tg-viewer-key-2026" http://localhost:8000/v1/audit/verify
```

---

## 4. Policy Configuration & Tuning

Governance policies are managed in `configs/policy.yaml`:

```yaml
version: "1.0.0"
default_risk_tier: "standard"

thresholds:
  damage_approve_max: 0.05       # Max predicted damage upper bound for direct approval (5%)
  damage_repair_max: 0.20        # Max predicted damage eligible for automated repair (20%)
  damage_reject_min: 0.20        # Damage above 20% is strictly quarantined
  conformal_confidence: 0.90     # Split-conformal coverage level (90%)
  max_repair_budget_pct: 0.15    # Maximum fraction of cells modified during repair (15%)

hard_blocks:
  unmasked_direct_pii: true      # Direct unmasked SSN, Aadhaar, PAN, CC triggers quarantine
  breaking_schema_change: true   # Missing required columns triggers quarantine
  corrupted_audit_hash: true     # Lineage failure triggers quarantine

human_review:
  escalate_on_ood: true          # Out-of-distribution batches escalate to operator
  escalate_on_boundary_margin: 0.01
```

---

## 5. Incident Response & Playbooks

### Incident 1: High Quarantine Rate Spike
**Symptoms**: Spike in `trustgate_batches_quarantined_total` metrics or dashboard alarm.
**Diagnosis**:
1. Check `GET /v1/batches?status=QUARANTINE`.
2. Review batch decision records:
   - If `hard_block_triggered == true`: Check `hard_block_reasons`. Common causes: upstream schema change or unmasked PII in ingest stream.
   - If `predicted_damage` high: Inspect top SHAP factor drivers.
3. If an upstream producer deployed an unannounced schema change, contact producer and update the corresponding `DataContract`.

### Incident 2: Cryptographic Audit Signature Mismatch
**Symptoms**: `verify` command exits with code 1: `AUDIT CHAIN COMPROMISED`.
**Diagnosis**:
1. Check the line number reported by `verify_audit_ledger`.
2. Inspect whether disk corruption, unauthorized edit, or process crash caused an incomplete write.
3. Restore ledger from verified offsite backup and compare block hashes.

### Incident 3: Out-of-Distribution Escalation Flood
**Symptoms**: Hundreds of batches queued in `JobStatus.ESCALATED_WAITING_FOR_REVIEW`.
**Diagnosis**:
1. Incoming data distribution has undergone structural shift (e.g. new market expansion, inflation, demographic change).
2. Human operators should review sample batches in the dashboard.
3. If the shift represents legitimate new business data, retrain target models and update the `DamagePredictor` baseline distribution (`make evaluate`).
