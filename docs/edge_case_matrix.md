# Edge Case Verification Matrix

This matrix tracks the mapping between all explicit edge cases specified in Section 8 of the TrustGate specification, their corresponding target module, the designated test name, and implementation status.

| ID | Category | Edge Case Description | Target Module | Designated Test Name | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **8.1.1** | Ingest | Empty file (0 bytes) | `trustgate.ingest.readers` | `test_ingest_empty_and_header_only_file` | Implemented (Phase 1) |
| **8.1.2** | Ingest | Header-only file (0 data rows) | `trustgate.ingest.readers` | `test_ingest_empty_and_header_only_file` | Implemented (Phase 1) |
| **8.1.3** | Ingest | Single row / single column table | `trustgate.ingest.readers` | `test_ingest_single_row_single_col` | Implemented (Phase 1) |
| **8.1.4** | Ingest | Ragged rows, mixed delimiters, quoted newlines | `trustgate.ingest.readers` | `read_batch_file` | Implemented (Phase 1) |
| **8.1.5** | Ingest | UTF-8 with BOM, Latin-1, Mojibake encodings | `trustgate.ingest.readers` | `test_ingest_encodings_and_bom` | Implemented (Phase 1) |
| **8.1.6** | Ingest | Duplicate, empty, whitespace-variant column names | `trustgate.ingest.sanitizer` | `test_sanitize_column_names_edge_cases` | Implemented (Phase 1) |
| **8.1.7** | Ingest | Injection strings in headers (`=cmd\|...`, `"Ignore instructions"`) | `trustgate.ingest.sanitizer` | `test_sanitize_column_names_edge_cases` | Implemented (Phase 1) |
| **8.1.8** | Ingest | Mixed types in column (numbers + strings) | `trustgate.ingest.schema` | `test_numeric_parsing_locales_and_symbols` | Implemented (Phase 1) |
| **8.1.9** | Ingest | Numeric strings with commas vs locale decimals (`1,234.5` vs `1.234,5`) | `trustgate.ingest.schema` | `test_numeric_parsing_locales_and_symbols` | Implemented (Phase 1) |
| **8.1.10** | Ingest | Multi-format dates, ambiguous day/month, epoch millis vs seconds | `trustgate.ingest.schema` | `test_boolean_and_timestamp_parsing` | Implemented (Phase 1) |
| **8.1.11** | Ingest | Boolean variants (Y/N, 0/1, True/False, yes/no) | `trustgate.ingest.schema` | `test_boolean_and_timestamp_parsing` | Implemented (Phase 1) |
| **8.1.12** | Ingest | Decompression bombs / ratio check (100:1 max ratio limit) | `trustgate.ingest.readers` | `test_decompression_bomb_detection` | Implemented (Phase 1) |
| **8.1.13** | Ingest | Zero-variance numeric columns (division by zero guards in PSI/z-score) | `trustgate.profiling.drift` | `test_drift_zero_variance_and_small_batch` | Implemented (Phase 1) |
| **8.2.1** | Contracts | No contract provided (automatic contract inference with `INFERRED` mark) | `trustgate.contracts.inferrer` | `test_contract_inference` | Implemented (Phase 1) |
| **8.2.2** | Contracts | Schema version mismatch, breaking vs non-breaking changes | `trustgate.contracts.validator` | `test_contract_breaking_missing_required_column` | Implemented (Phase 1) |
| **8.2.3** | Contracts | Malicious regex / ReDoS in contract definition | `trustgate.contracts.validator` | `test_safe_regex_match_and_redos_guard` | Implemented (Phase 1) |
| **8.3.1** | Stats/Drift | Small batches (<30 rows): fallback to permutation/exact tests or `INSUFFICIENT_DATA` | `trustgate.profiling.drift` | `test_drift_zero_variance_and_small_batch` | Implemented (Phase 1) |
| **8.3.2** | Stats/Drift | Large batches: effect sizes (Cramér's V, PSI) paired with p-values | `trustgate.profiling.drift` | `test_categorical_drift_and_new_categories` | Implemented (Phase 1) |
| **8.3.3** | Stats/Drift | PSI with zero-count bins ($\epsilon$-smoothing) and quantile ties | `trustgate.profiling.drift` | `test_psi_identical_and_drifted` | Implemented (Phase 1) |
| **8.3.4** | Stats/Drift | Multiple testing correction (Benjamini-Hochberg FDR) across columns | `trustgate.profiling.drift` | `test_fdr_correction_multi_column` | Implemented (Phase 1) |
| **8.3.5** | Stats/Drift | Extreme class imbalance (0.1% positives) and single-class batches | `trustgate.profiling.label_profiler` | `test_labeled_batch_single_class_edge_case` | Implemented (Phase 1) |
| **8.4.1** | Simulator | MCAR, MAR, MNAR missingness injection | `trustgate.simulator.injectors` | `test_missingness_injectors` | Implemented (Phase 3) |
| **8.4.2** | Simulator | Covariate drift (mean, variance, unit change cm to m, unseen categories) | `trustgate.simulator.injectors` | `test_covariate_drift_injectors` | Implemented (Phase 3) |
| **8.4.3** | Simulator | Concept drift & label noise (symmetric, asymmetric, instance-dependent) | `trustgate.simulator.injectors` | `test_concept_drift_and_label_noise` | Implemented (Phase 3) |
| **8.4.4** | Simulator | Compound corruptions (2-4 simultaneous interacting issues) | `trustgate.simulator.compound` | `test_compound_corruption` | Implemented (Phase 3) |
| **8.5.1** | PII | Indian Aadhaar (12-digit with Verhoeff checksum) detection & rejection | `trustgate.pii.detectors` | `test_pii_aadhaar_verhoeff_checksum` | Implemented (Phase 2) |
| **8.5.2** | PII | Indian PAN (`AAAAA9999A`) and Indian mobile phone formats (`+91`) | `trustgate.pii.detectors` | `test_pii_pan_and_indian_phones` | Implemented (Phase 2) |
| **8.5.3** | PII | International PII (Credit cards via Luhn, Passports, Emails, UPI, IBAN) | `trustgate.pii.detectors` | `test_pii_international_types` | Implemented (Phase 2) |
| **8.5.4** | PII | Irreversible masking & salted HMAC tokenization | `trustgate.pii.masking` | `test_pii_masking_and_hmac_tokenization` | Implemented (Phase 2) |
| **8.5.5** | PII | Zero raw cell values sent to LLM prompt | `trustgate.agents.guardrails` | `test_guardrails_prompt_injection_defense` | Implemented (Phase 6) |
| **8.6.1** | Repair | Repair worsens data (post-repair damage > pre-repair): rollback to quarantine | `trustgate.repair.planner` | `test_closed_loop_budget_cap_enforcement` | Implemented (Phase 5) |
| **8.6.2** | Repair | Reference-fitted imputation (forbidden to fit on corrupted batch) | `trustgate.repair.strategies` | `test_reference_fitted_imputation` | Implemented (Phase 5) |
| **8.6.3** | Repair | Idempotency ($Repair(Repair(X)) == Repair(X)$) | `trustgate.repair.strategies` | `test_repair_idempotency` | Implemented (Phase 5) |
| **8.6.4** | Repair | Repair budget exceeded (cells changed > cap): reject to quarantine | `trustgate.repair.planner` | `test_closed_loop_budget_cap_enforcement` | Implemented (Phase 5) |
| **8.7.1** | Agents/LLM | LLM failure/timeout/5xx: seamless fallback to deterministic template explainer | `trustgate.agents.llm_clients` | `test_template_llm_client_offline` | Implemented (Phase 6) |
| **8.7.2** | Agents/LLM | Prompt injection in data values/headers: sanitized and non-influential on Judge | `trustgate.agents.guardrails` | `test_guardrails_prompt_injection_defense` | Implemented (Phase 6) |
| **8.7.3** | Agents/LLM | Numeric claim verification: rejected hallucinated numbers in explanations | `trustgate.agents.verifier` | `test_numeric_claim_verifier_hallucination_trap` | Implemented (Phase 6) |
| **8.8.1** | Policy | Deterministic policy evaluation & separation of duties | `trustgate.policy.engine` | `test_policy_decision_clean_batch_approve` | Implemented (Phase 5) |
| **8.9.1** | Audit | Hash-chained log tampering detection via `verify` | `trustgate.audit.ledger` | `test_audit_hash_chain_tamper_detection` | Planned (Phase 7) |
