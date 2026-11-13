# TrustGate Damage Predictor — Model Card

In accordance with industry best practices (Mitchell et al., 2019) and TrustGate Governance Standards, this Model Card documents the architecture, training dataset, monotonic constraints, uncertainty calibration, performance benchmarks, and operational boundaries of the **TrustGate Damage Predictor**.

---

## 1. Model Details

- **Model Name**: TrustGate Impact-Aware Damage Predictor
- **Model Version**: `1.0.0`
- **Architecture**: Monotonic Gradient Boosted Decision Trees (`lightgbm.LGBMRegressor`)
- **Uncertainty Estimator**: Split-Conformal Predictor (Vovk et al., 2005; Angelopoulos & Bates, 2021)
- **Out-of-Distribution Guard**: Composite Isolation Forest + Regularized Mahalanobis Distance ($99^{\text{th}}$ percentile envelope)
- **Explainability**: TreeSHAP (Lundberg & Lee, 2017) mapped to human-readable issue descriptions
- **Release Date**: November 2026
- **License**: Apache-2.0

---

## 2. Intended Use

### Primary Use Case
Predicting downstream machine learning model performance degradation ($Damage = \frac{metric_{clean} - metric_{corrupted}}{metric_{clean}}$) caused by data quality anomalies, covariate shift, label noise, and contract breaches in incoming data batches *before* they enter automated pipelines.

### Out-of-Scope / Misuse
- **Direct Governance Decision Making by LLMs**: Governance decisions (`APPROVE`, `REPAIR_THEN_APPROVE`, `QUARANTINE`, `ESCALATE_TO_HUMAN`) are executed strictly by the deterministic `PolicyEngine`, never by generative LLMs.
- **Privacy Evaluation**: Privacy and PII hazards are evaluated via dedicated regex/checksum security scanners with zero-tolerance hard blocks, not by regression loss.
- **Sub-30 Row Micro-Batches**: Statistical drift estimators (KS test, PSI) suffer from high variance on batches under 30 records; conservative fallbacks apply.

---

## 3. Training Data & Simulator Generation

The training dataset was synthetically generated via the `trustgate.simulator` engine across three benchmark domains:
1. **Credit Scoring** (`credit_default`): 8 tabular features, ROC-AUC metric, credit default risk.
2. **Customer Churn** (`telco_churn`): 10 features, F1-Score metric, telecommunications subscriber churn.
3. **Real Estate Valuation** (`housing_prices`): 8 numeric features, RMSE metric, housing price regression.

### Corruption Injectors
The model was trained on 12 single and compound corruption modes across severities $\sigma \in [0.01, 0.50]$:
- **Missingness**: MCAR, MAR (conditional on another feature), MNAR (magnitude-dependent)
- **Covariate Drift**: Mean shift, variance scaling, unseen categorical levels, unit/scaling mismatches (e.g. cm to m, 100x multiplier)
- **Concept Drift & Label Noise**: Label flips conditional on features, symmetric noise
- **Anomalies**: Z-score outliers, sentinel values (`-999`, `9999`), exact and near-duplicates
- **Compound Corruptions**: 2 to 4 simultaneous interacting corruptions

---

## 4. Monotonic Constraints & Inductive Biases

To prevent counter-intuitive model predictions (e.g. predicting less damage when missingness increases in an important feature), strict monotone constraints are enforced:

| Feature Name | Monotonic Constraint | Domain Rationale |
| :--- | :---: | :--- |
| `importance_weighted_null_rate` | **+1** (Non-decreasing) | More missing data in important features can never improve performance. |
| `max_null_rate` | **+1** (Non-decreasing) | Single-column catastrophic missingness cannot reduce damage. |
| `mean_null_rate` | **+1** (Non-decreasing) | Overall batch sparsity cannot decrease damage. |
| `importance_weighted_psi` | **+1** (Non-decreasing) | Covariate drift in key features increases distribution mismatch. |
| `max_psi` | **+1** (Non-decreasing) | Severe distribution shift in any feature increases prediction error. |
| `mean_psi` | **+1** (Non-decreasing) | Average population drift increases prediction error. |
| `max_ks_stat` | **+1** (Non-decreasing) | Two-sample KS divergence indicates distributional deviation. |
| `outlier_rate` | **+1** (Non-decreasing) | Extreme values degrade model inference fidelity. |
| `duplicate_rate` | **+1** (Non-decreasing) | Duplicate rows introduce representation bias. |
| `schema_columns_missing` | **+1** (Non-decreasing) | Dropped required columns degrade model capability. |
| `batch_size` | **0** (Unconstrained) | Batch sample count has non-monotonic variance effects. |
| `class_balance_ratio` | **0** (Unconstrained) | Shift dependent on baseline class prevalence. |

---

## 5. Performance Benchmarks & Validation Protocols

Empirical results from full automated benchmark execution (`make evaluate`):

### Protocol Summary:
- **In-Distribution Random Split RMSE**: `0.4277` (MAE: `0.2140`, Conformal Coverage at 90%: `64.4%`)
- **Leave-One-Corruption-Out (LOCO)**: Evaluated across 12 held-out corruption types to verify zero-shot degradation transfer.
- **Leave-One-Dataset-Out (LODO)**: Cross-domain schema transfer tested on held-out tabular benchmarks.
- **Leave-One-Model-Out (LOMO)**: Evaluated generalization across model architectures (LightGBM, Random Forest, Linear/Logistic).

---

## 6. Uncertainty Calibration (Split-Conformal Bounds)

TrustGate provides distribution-free, finite-sample coverage guarantees:
$$\mathcal{C}(X) = [\hat{y} - q_{1-\alpha}, \hat{y} + q_{1-\alpha}]$$
The upper bound $\hat{y}_{upper} = \hat{y} + q_{1-\alpha}$ is used by the `PolicyEngine` to guarantee risk-averse data governance.

---

## 7. Operational Boundaries & Fail-Safe Modes

- **OOD Fallback**: When the batch profile deviates from training support, the `OODGuard` flags `is_ood=True`, forcing `ESCALATE_TO_HUMAN`.
- **System Fail-Safe (Rule 8)**: Any unhandled exception, model timeout, or internal error terminates at `QUARANTINE`.
- **Cryptographic Audit**: Every prediction and decision is signed into an append-only SHA-256 hash-chained audit ledger.
