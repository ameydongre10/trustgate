# Data Governance, Risk Management & AI-Readiness Alignment

## 1. Governance Alignment Overview

TrustGate is designed to operationalize data governance, enterprise risk management (ERM), and AI safety frameworks directly within CI/CD and production streaming pipelines. 

> **Important Notice**: TrustGate provides architectural alignment and automated controls to support model risk management and data governance practices. It does not constitute formal legal compliance or certification under statutory regulations.

---

## 2. Regulatory & Framework Alignment

### 2.1 US Federal Reserve SR 11-7 / OCC 2011-12 (Supervisory Guidance on Model Risk Management)
- **Conceptual Soundness**: TrustGate verifies that input data distributions match the assumptions and training references of the model (`TargetModel` registration), providing quantitative evidence of data suitability.
- **Ongoing Monitoring**: Continuous statistical drift profiling (KS-test, PSI, Cramér's V) provides rigorous statistical tracking of covariate and concept drift.
- **Outcomes Analysis**: By predicting downstream metric degradation using conformal prediction intervals, TrustGate quantifies empirical model risk before corrupt data degrades downstream decisions.

### 2.2 EU Artificial Intelligence Act (Data Quality & Technical Documentation - Article 10 & 11)
- **Data Governance & Management Practices**: Enforces data contracts covering schema, nullability, statistical bounds, and demographic distributions.
- **Traceability & Audit Logging**: Cryptographically hash-chained append-only logs document every batch's input hash, profile metrics, policy decisions, and lineage modifications.
- **Human Oversight**: Mandatory human-in-the-loop escalation paths for high-risk, boundary, or out-of-distribution batches.

### 2.3 Privacy Regulations (GDPR, DPDP India, HIPAA, CCPA)
- **Data Minimization & PII Gating**: Automated detection of Indian identifiers (Aadhaar with Verhoeff checksum, PAN), international identities (Credit Cards via Luhn, Passports, IBAN, Phone numbers, Emails).
- **Hard-Block Policies**: Batches containing unmasked, uncontracted direct identifiers are automatically blocked and routed to an isolated quarantine store.
- **Irreversible Masking & Tokenization**: Salted HMAC tokenization prevents exposure of plain identifiers while preserving referential integrity for analytical joins.
- **Right to Erasure (RTBF)**: Quarantine and artifact storage support targeted tombstoning and deletion hooks without compromising the mathematical hash continuity of the audit ledger.

---

## 3. Separation of Duties and Operational Controls

1. **Submitter vs. Approver**: A batch submitter cannot unilaterally approve an escalated or quarantined batch. Approvals require designated `approver` or `admin` credentials.
2. **Deterministic Judge Policy**: The decision logic is governed by declarative code in `configs/policy.yaml`. No LLM possesses authority to grant waivers or alter policy gates.
3. **Emergency Override Protocol**: Emergency overrides require an explicit justification, are time-bounded, and trigger specialized high-severity audit logging events.
