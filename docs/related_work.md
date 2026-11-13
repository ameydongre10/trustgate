# Related Work: Data Validation, Drift Monitoring, and Impact-Aware Data Trust

## 1. Executive Summary

Data quality (DQ) validation and machine learning (ML) observability have traditionally operated as disjoint disciplines:
1. **Rule-based data quality systems** (e.g., Great Expectations, Deequ, Soda Core, Pandera) validate tabular datasets against human-authored assertions (null bounds, schema types, regex formats).
2. **ML drift and observability monitors** (e.g., Evidently AI, WhyLogs, Arize, Fiddler) track population drift (Kolmogorov-Smirnov test, Population Stability Index, Wasserstein distance) post-hoc or at batch boundaries.
3. **Academic impact-driven data cleaning** (e.g., ActiveClean, BoostClean, CPClean) explores how cleaning noisy labels or missing values improves downstream convex optimization or classification benchmarks.

**TrustGate** bridges the gap between raw data validation and operational ML risk management. Rather than failing batches based on arbitrary counts of rule violations or isolated p-values, TrustGate computes a model-aware *Issue Profile* (weighted by the target model's empirical feature importances) and uses a learned, conformalized *Damage Predictor* to estimate the exact downstream metric degradation (e.g., ROC-AUC drop or RMSE increase) before deciding to Approve, Repair, Quarantine, or Escalate.

---

## 2. Taxonomy of Existing Systems and Literature

### 2.1 Rule-Based Data Validation Engines
- **Great Expectations (GX)**: Industry standard for declarative assertions ("Expectations"). Evaluates batch data against pre-compiled expectation suites, producing HTML Data Docs.
  - *Limitation*: Rule violations are unweighted and uncalibrated against downstream ML tasks. A missing column in an unused auxiliary feature causes the same pipeline failure as a missing primary signal.
- **AWS Deequ / PyDeequ**: Computes unit tests for data on Apache Spark using declarative constraints (e.g., `hasCompleteness`, `hasCorrelation`).
  - *Limitation*: Requires heavy Spark infrastructure and treats all constraint breaches as equally severe unless manually tuned with custom heuristics.
- **Soda Core & SodaCL**: Human-readable YAML DSL for data testing across SQL databases.
  - *Limitation*: Focuses on analytics tables and warehousing; no awareness of ML model sensitivity, feature interactions, or predictive damage.
- **Pandera**: Type-system and statistical schema validation for Pandas/Polars DataFrames.
  - *Limitation*: Validates column dtypes and value ranges, but lacks automated impact estimation and multi-step repair workflows.

### 2.2 ML Observability and Drift Monitoring
- **Evidently AI**: Open-source library providing interactive reports and metric calculations for data drift (KS-test, Wasserstein, PSI, Jensen-Shannon divergence), model quality, and target drift.
  - *Limitation*: Primarily diagnostic/monitoring. It signals that drift has occurred, but does not provide an automated closed-loop policy engine to estimate downstream damage or execute verifiable repairs.
- **WhyLogs / WhyLabs**: Lightweight statistical profiling using streaming sketches (KLL sketches, frequent items, HLL).
  - *Limitation*: Excellent for large-scale distributed telemetry, but leaves gating policies to arbitrary manual thresholds.
- **Commercial Observability (Arize, Fiddler, Monte Carlo)**: Focus on production inference telemetry, root-cause analysis, and data downtime tracking.

### 2.3 Academic Research: Data Quality Impact on Machine Learning
- **ActiveClean & BoostClean (Krishnan et al., VLDB 2016, 2017)**: Iteratively cleans dirty training data to minimize validation loss of convex machine learning models via gradient-based updates.
  - *Relation to TrustGate*: Demonstrates that not all dirty data impacts model convergence equally. However, ActiveClean is designed for interactive training iterations, whereas TrustGate focuses on batch-level inference and retraining gating.
- **CPClean (Karlas et al., SIGMOD 2020)**: Evaluates the impact of dirty data on predictions using Certified Predictions with nearest-neighbor and tree ensembles.
- **TFX / TensorFlow Data Validation (TFDV) (Baylor et al., KDD 2017)**: Analyzes training and serving data at scale to detect skew and schema anomalies.
- **Conformal Prediction (Vovk et al. 2005; Angelopoulos & Bates 2021)**: Finite-sample, distribution-free uncertainty quantification guaranteeing that prediction intervals contain the ground truth with a user-specified probability (e.g., 90%).

---

## 3. Honest Assessment: What is Novel vs. What is NOT Novel

To ensure intellectual rigor and transparency, TrustGate explicitly clarifies its technical boundaries:

### What is NOT Novel in TrustGate
1. **Statistical Drift Metrics**: TrustGate uses established, standard statistical tests (two-sample Kolmogorov-Smirnov test, Population Stability Index with epsilon-smoothed binning, Chi-square test of independence, Jensen-Shannon divergence, and Cramér's V). These are mathematically standard.
2. **Synthetic Corruption Injection**: Generating artificial missingness (MCAR, MAR, MNAR), Gaussian noise, label flips, and categorical shifts is standard methodology in ML robustness benchmarking.
3. **Regex and Algorithmic PII Identification**: Regular expressions and checksum algorithms (Luhn for Credit Cards, Verhoeff for Aadhaar) are industry-standard patterns.
4. **Split-Conformal Prediction**: Conformal intervals calculated via calibration non-conformity scores follow the established formulation of distribution-free uncertainty estimation.

### What IS Novel in TrustGate
1. **Downstream Impact-Driven Gating Architecture**: Unlike tools that count rule failures, TrustGate translates dirty batch characteristics into a standardized *Issue Profile*, weights anomalies by the target model's pre-computed feature importances, and passes them to a trained *Damage Predictor*. The decision engine evaluates the *conformal upper bound* of predicted performance loss ($Damage_{upper}$) against policy thresholds.
2. **Closed-Loop "Repair-then-Verify" Gating Loop**: When a repair strategy (e.g., median reference imputation, winsorization, or schema realignment) is planned, the system does not assume success. It re-profiles the repaired batch, re-runs damage prediction, and ensures that the post-repair predicted damage is below tolerance *without introducing secondary corruptions or distorting reference distributions*.
3. **Cryptographically Tamper-Evident Hash-Chained Audit Trail**: Integrating append-only SHA-256 hash chaining across batch metadata, issue profiles, conformal predictions, policy decisions, and lineage patches into a standalone, verifiable log.
4. **Deterministic Judge with Explainability Guardrails**: Strictly isolating policy decisions from non-deterministic LLM behavior. LLMs only provide natural-language translations, while a deterministic *Numeric-Claim Verifier* strictly prevents hallucinations by verifying all reported figures against the underlying decision record.

---

## 4. References

1. Krishnan, S., Wang, J., Franklin, M. J., Goldberg, K., & Kraska, T. (2016). *ActiveClean: Interactive Data Cleaning for Statistical Modeling*. Proceedings of the VLDB Endowment, 9(12), 948-959.
2. Krishnan, S., Tu, Z., Franklin, M. J., & Goldberg, K. (2017). *BoostClean: Automated Error Detection and Repair for Machine Learning*. arXiv:1711.01299.
3. Schelter, S., Lange, D., Schmidt, P., Celikel, M., Biessmann, F., & Grafberger, A. (2018). *Automating large-scale data quality verification*. Proceedings of the VLDB Endowment, 11(12), 1783-1794.
4. Baylor, D., et al. (2017). *TFX: A TensorFlow-Based Production-Scale Machine Learning Platform*. KDD '17, 1387-1395.
5. Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.
6. Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). John Wiley & Sons.
