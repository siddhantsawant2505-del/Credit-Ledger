# Home Credit Default Risk - Data Quality & Loading Report

> **Generated at**: 2026-09-27 14:53:55
> **Data Directory**: `E:\Projects\Credit-Ledger\data`

## 0. Data Authenticity Check

- **Verdict**: `:x: SYNTHETIC_OR_TRUNCATED`
- [WARN] Loaded data does NOT match the real Kaggle Home Credit dataset scale. These shapes match server/generate_sample_data.py output. Preprocessing/training on this data will NOT produce valid results. Run 'python server/fetch_data.py' to download the real dataset into data/.

| Table | Actual Shape | Expected Minimum (rows, cols) | Status |
| :--- | :--- | :--- | :--- |
| `application_train` | 10,000 x 17 | 300,000 x 122 | TOO SMALL (synthetic/truncated) |
| `application_test` | 2,000 x 16 | 48,000 x 121 | TOO SMALL (synthetic/truncated) |
| `bureau` | 15,000 x 8 | 1,600,000 x 17 | TOO SMALL (synthetic/truncated) |
| `bureau_balance` | 25,000 x 3 | 27,000,000 x 3 | TOO SMALL (synthetic/truncated) |
| `previous_application` | 10,000 x 6 | 1,600,000 x 37 | TOO SMALL (synthetic/truncated) |
| `POS_CASH_balance` | 15,000 x 7 | 10,000,000 x 8 | TOO SMALL (synthetic/truncated) |
| `installments_payments` | 15,000 x 8 | 13,600,000 x 8 | TOO SMALL (synthetic/truncated) |
| `credit_card_balance` | 8,000 x 7 | 3,800,000 x 8 | TOO SMALL (synthetic/truncated) |

## 1. Dataset Overview & Memory Footprint

| Table | Rows | Columns | Initial Memory | Downcast Memory | Memory Saved |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `application_train` | 10,000 | 17 | 2.99 MB | 0.44 MB | 2.55 MB (-85.3%) |
| `application_test` | 2,000 | 16 | 0.58 MB | 0.09 MB | 0.50 MB (-85.1%) |
| `bureau` | 15,000 | 8 | 1.59 MB | 0.34 MB | 1.24 MB (-78.4%) |
| `bureau_balance` | 25,000 | 3 | 1.57 MB | 0.14 MB | 1.43 MB (-90.9%) |
| `previous_application` | 10,000 | 6 | 1.44 MB | 0.17 MB | 1.27 MB (-88.0%) |
| `POS_CASH_balance` | 15,000 | 7 | 1.47 MB | 0.19 MB | 1.29 MB (-87.4%) |
| `installments_payments` | 15,000 | 8 | 0.92 MB | 0.31 MB | 0.60 MB (-65.6%) |
| `credit_card_balance` | 8,000 | 7 | 0.43 MB | 0.19 MB | 0.24 MB (-55.3%) |
| **TOTAL** | **100,000** | - | **10.99 MB** | **1.88 MB** | **9.11 MB (-82.9%)** |

### Categorical Columns (created during downcast)

- `application_train`: 4 categorical column(s) - `NAME_CONTRACT_TYPE`, `CODE_GENDER`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`
- `application_test`: 4 categorical column(s) - `NAME_CONTRACT_TYPE`, `CODE_GENDER`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`
- `bureau`: 1 categorical column(s) - `CREDIT_ACTIVE`
- `bureau_balance`: 1 categorical column(s) - `STATUS`
- `previous_application`: 2 categorical column(s) - `NAME_CONTRACT_TYPE`, `NAME_CONTRACT_STATUS`
- `POS_CASH_balance`: 1 categorical column(s) - `NAME_CONTRACT_STATUS`

## 2. Main Table (`application_train`) Profiling

- **Shape**: `10,000` rows x `17` columns
- **Total Missing Cells**: `14,500` (8.53% of all cells)
- **Columns with Missing Values**: `4` / `17`

### Data Types Distribution
- `float32`: 8 columns
- `category`: 1 columns
- `int8`: 2 columns
- `int16`: 2 columns
- `int32`: 1 columns

### TARGET Class Balance
- **Class 0 (Repaid / Non-Default)**: 9,209 (92.09%)
- **Class 1 (Defaulted / Payment Difficulties)**: 791 (7.91%)
- **Class Imbalance Ratio**: ~ `11.64:1` (severe class imbalance)

### Top 4 Missing Columns
| Column | Missing Count | Missing % | Dtype |
| :--- | :--- | :--- | :--- |
| `OWN_CAR_AGE` | 6,500 | 65.00% | `float32` |
| `EXT_SOURCE_1` | 5,500 | 55.00% | `float32` |
| `EXT_SOURCE_3` | 2,000 | 20.00% | `float32` |
| `AMT_ANNUITY` | 500 | 5.00% | `float32` |

## 3. Join Keys & Cardinality Validation

Before merging, all primary and foreign key constraints are validated:

| Table | Join Key | Role | Status | Rows | Nulls (%) | Unique Keys | Cardinality Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `application_train` | `SK_ID_CURR` | Parent | PASS (Valid) | 10,000 | 0 (0.0%) | 10,000 | 1:1 (Unique) |
| `application_test` | `SK_ID_CURR` | Parent | PASS (Valid) | 2,000 | 0 (0.0%) | 2,000 | 1:1 (Unique) |
| `bureau` | `SK_ID_CURR` | Child | PASS (Valid) | 15,000 | 0 (0.0%) | 8,539 | 1:N (Avg 1.8 rows/key) |
| `previous_application` | `SK_ID_CURR` | Child | PASS (Valid) | 10,000 | 0 (0.0%) | 6,765 | 1:N (Avg 1.5 rows/key) |
| `POS_CASH_balance` | `SK_ID_CURR` | Child | PASS (Valid) | 15,000 | 0 (0.0%) | 5,726 | 1:N (Avg 2.6 rows/key) |
| `installments_payments` | `SK_ID_CURR` | Child | PASS (Valid) | 15,000 | 0 (0.0%) | 5,705 | 1:N (Avg 2.6 rows/key) |
| `credit_card_balance` | `SK_ID_CURR` | Child | PASS (Valid) | 8,000 | 0 (0.0%) | 4,397 | 1:N (Avg 1.8 rows/key) |
| `previous_application` | `SK_ID_PREV` | Parent | PASS (Valid) | 10,000 | 0 (0.0%) | 10,000 | 1:1 (Unique) |
| `POS_CASH_balance` | `SK_ID_PREV` | Child | PASS (Valid) | 15,000 | 0 (0.0%) | 7,749 | 1:N (Avg 1.9 rows/key) |
| `installments_payments` | `SK_ID_PREV` | Child | PASS (Valid) | 15,000 | 0 (0.0%) | 7,789 | 1:N (Avg 1.9 rows/key) |
| `credit_card_balance` | `SK_ID_PREV` | Child | PASS (Valid) | 8,000 | 0 (0.0%) | 5,523 | 1:N (Avg 1.4 rows/key) |
| `bureau` | `SK_ID_BUREAU` | Parent | PASS (Valid) | 15,000 | 0 (0.0%) | 15,000 | 1:1 (Unique) |
| `bureau_balance` | `SK_ID_BUREAU` | Child | PASS (Valid) | 25,000 | 0 (0.0%) | 12,133 | 1:N (Avg 2.1 rows/key) |

### Cross-Table Referential Integrity (Orphan Record Check)

| Join Key | Parent Table(s) | Child Table | Unique Keys in Child | Orphan Keys in Child | Integrity Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `SK_ID_CURR` | `application_train + test` | `bureau` | 8,539 | 0 (0.0%) | PASS (100% matched) |
| `SK_ID_CURR` | `application_train + test` | `previous_application` | 6,765 | 0 (0.0%) | PASS (100% matched) |
| `SK_ID_CURR` | `application_train + test` | `POS_CASH_balance` | 5,726 | 0 (0.0%) | PASS (100% matched) |
| `SK_ID_CURR` | `application_train + test` | `installments_payments` | 5,705 | 0 (0.0%) | PASS (100% matched) |
| `SK_ID_CURR` | `application_train + test` | `credit_card_balance` | 4,397 | 0 (0.0%) | PASS (100% matched) |
| `SK_ID_PREV` | `previous_application` | `POS_CASH_balance` | 7,749 | 0 (0.0%) | PASS (100% matched) |
| `SK_ID_PREV` | `previous_application` | `installments_payments` | 7,789 | 0 (0.0%) | PASS (100% matched) |
| `SK_ID_PREV` | `previous_application` | `credit_card_balance` | 5,523 | 0 (0.0%) | PASS (100% matched) |
| `SK_ID_BUREAU` | `bureau` | `bureau_balance` | 12,133 | 0 (0.0%) | PASS (100% matched) |

## 4. Warnings & Anomalies Detected

- [CRITICAL] Loaded data does NOT match the real Kaggle Home Credit dataset scale. These shapes match server/generate_sample_data.py output. Preprocessing/training on this data will NOT produce valid results. Run 'python server/fetch_data.py' to download the real dataset into data/.
- [PASS] **No critical anomalies or key mismatches detected.** All foreign keys cleanly link to parent tables.

---

*Report generated by `server/data_loader.py` - :warning: Data NOT ready for preprocessing: replace synthetic/truncated files with the real Kaggle dataset (`python server/fetch_data.py`).*