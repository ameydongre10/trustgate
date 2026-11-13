# TrustGate PII Detection Benchmark Report

## 1. Summary Metrics

- **Total Benchmark Samples**: 1200
- **Overall Precision**: **1.0000** (100.00%)
- **Overall Recall**: **1.0000** (100.00%)
- **Overall F1-Score**: **1.0000**
- **False Positives**: 0 (out of 600 negative controls)
- **False Negatives**: 0 (out of 600 positive PII instances)

## 2. Granular Breakdown by Identifier Category

| Category / Identifier | Test Type | TP | FP | FN | TN | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `credit_card` | Positive PII | 100 | 0 | 0 | 0 | 100.00% | 100.00% | 1.00 |
| `email` | Positive PII | 100 | 0 | 0 | 0 | 100.00% | 100.00% | 1.00 |
| `indian_aadhaar` | Positive PII | 100 | 0 | 0 | 0 | 100.00% | 100.00% | 1.00 |
| `indian_pan` | Positive PII | 100 | 0 | 0 | 0 | 100.00% | 100.00% | 1.00 |
| `indian_phone` | Positive PII | 100 | 0 | 0 | 0 | 100.00% | 100.00% | 1.00 |
| `non_pii_12digit` | Negative Control | 0 | 0 | 0 | 100 | 100.00% | 100.00% | 1.00 |
| `non_pii_16digit` | Negative Control | 0 | 0 | 0 | 100 | 100.00% | 100.00% | 1.00 |
| `non_pii_alphanumeric` | Negative Control | 0 | 0 | 0 | 100 | 100.00% | 100.00% | 1.00 |
| `non_pii_order_id` | Negative Control | 0 | 0 | 0 | 100 | 100.00% | 100.00% | 1.00 |
| `non_pii_text` | Negative Control | 0 | 0 | 0 | 100 | 100.00% | 100.00% | 1.00 |
| `non_pii_url` | Negative Control | 0 | 0 | 0 | 100 | 100.00% | 100.00% | 1.00 |
| `upi_id` | Positive PII | 100 | 0 | 0 | 0 | 100.00% | 100.00% | 1.00 |

## 3. Engineering Analysis
1. **Verhoeff & Luhn Checksum Filtering**: Zero false positives were observed on 12-digit and 16-digit negative controls due to mathematical checksum verification.
2. **Indian Identity Protection**: Aadhaar (12-digit UID) and PAN (`AAAAA9999A`) achieved 100% precision and recall on synthetic benchmark vectors.
3. **Order ID & Serial Number Discrimination**: Formats such as `ORDER-123456` are strictly separated from contact numbers and financial identifiers.