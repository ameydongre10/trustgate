# Architectural Decision Records (ADR) - TrustGate

## ADR-001: Technology Stack and Runtime
- **Date**: 2026-11-13
- **Status**: Accepted
- **Context**: TrustGate requires high-throughput data processing, deterministic ML training and evaluation, and fast, secure API execution.
- **Decision**:
  - Python 3.11 runtime.
  - Dependency management via `uv` (with `pyproject.toml` and pinned lockfile).
  - Storage & Cache optimization: Configure cache and large temp buffers to drive `D:\` (where 100+ GB is available) to avoid resource exhaustion on constrained OS partitions.
  - Core data engines: `polars` and `pyarrow` for memory-efficient chunked and lazy tabular operations; `pandas` exclusively at sklearn estimator boundaries.
- **Consequences**: Fast dependency resolution, resilient execution even on systems with constrained system drives, and sub-second batch validation for 1M+ row files.

## ADR-002: Deterministic Judge and Separation of LLM Concerns
- **Date**: 2026-11-13
- **Status**: Accepted
- **Context**: LLMs exhibit non-determinism, hallucinations, latency unpredictability, and susceptibility to prompt injection attacks.
- **Decision**:
  - The decision engine (`Judge`) is strictly deterministic. It operates entirely on quantitative policies, conformal uncertainty bounds, and explicit risk rules (`configs/policy.yaml`).
  - LLMs are utilized strictly for the `Explainer Agent` (translating structured decision records into natural language for various stakeholders).
  - All numbers emitted by the LLM are checked by a deterministic `NumericClaimVerifier` before presentation. If any claim is unverified or hallucinated, the system falls back to a deterministic, offline template explainer.
- **Consequences**: Zero regulatory or operational risk of hallucinated approvals or non-deterministic compliance decisions. Fully testable without active network or LLM API keys.

## ADR-003: Model-Aware Damage Prediction vs. Counting Rule Violations
- **Date**: 2026-11-13
- **Status**: Accepted
- **Context**: Traditional data quality tools fail pipelines when any rule is breached, causing severe "alert fatigue" and unnecessary blocking of resilient downstream models.
- **Decision**:
  - Register downstream consumers as `TargetModel` specifications (feature importances, reference distributions, performance metric).
  - Weight data anomalies (missingness, drift, outliers) by the target model's empirical feature importances.
  - Predict downstream degradation (`damage`) using a calibrated, monotonic gradient-boosted regressor equipped with split-conformal prediction intervals.
- **Consequences**: True risk-calibrated decisions: critical pipelines are protected from harmful data corruption while benign anomalies in low-importance features do not halt production pipelines.

## ADR-004: Closed-Loop Repair Verification
- **Date**: 2026-11-13
- **Status**: Accepted
- **Context**: Blindly applying data imputation or outlier clipping can introduce artificial distribution shift or alter feature relationships.
- **Decision**:
  - Implement a mandatory "Repair-then-Verify" loop. Any batch subjected to repair strategies (median imputation, winsorization, type coercion) is automatically re-profiled, re-checked for contract validity, and re-evaluated by the Damage Predictor.
  - Approvals are only granted if post-repair predicted damage is below threshold and no new issues are introduced.
- **Consequences**: Prevents "cure worse than disease" defects where data cleaning corrupts downstream predictions.

## ADR-005: Cryptographic Append-Only Audit Trail
- **Date**: 2026-11-13
- **Status**: Accepted
- **Context**: Enterprise governance and regulatory compliance (e.g., EU AI Act, Basel / model risk frameworks) require tamper-evident provenance of data decisions.
- **Decision**:
  - Implement an append-only JSONL log where each entry contains SHA-256 cryptographic hashes of the current record and the preceding record (`prev_hash`).
  - Provide a standalone CLI and API endpoint (`trustgate audit verify`) to prove provenance and detect any unauthorized manual alterations.
- **Consequences**: Mathematical auditability and immediate detection of retroactive log manipulation.

## ADR-006: Native TreeSHAP via LightGBM
- **Date**: 2026-11-13
- **Status**: Accepted
- **Context**: The `shap` package relies on `numba` and `llvmlite` JIT compilers, which trigger security/AppLocker blocking on locked Windows developer environments.
- **Decision**:
  - Leverage LightGBM's native C++ implementation of TreeSHAP (`predict(..., pred_contrib=True)`).
  - This computes mathematically exact TreeSHAP feature attributions and base values directly in C++ without external Python JIT compiler dependencies.
- **Consequences**: Exact TreeSHAP explanations with 10x lower latency, zero runtime JIT compilation overhead, and full portability across restricted operating environments.
