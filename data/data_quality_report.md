# Home Credit Default Risk - Data Quality & Loading Report

> **Generated at**: 2026-09-27 15:29:56
> **Data Directory**: `E:\Projects\Credit-Ledger\data`

## 0. Data Authenticity Check

- **Verdict**: `AUTHENTIC` - [OK] All loaded tables match the real Kaggle dataset scale. Safe to proceed with preprocessing.

## 1. Dataset Overview & Memory Footprint

| Table | Rows | Columns | Initial Memory | Downcast Memory | Memory Saved |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `application_train` | 307,511 | 122 | 504.99 MB | 95.32 MB | 409.66 MB (-81.1%) |
| `application_test` | 48,744 | 121 | 79.69 MB | 15.07 MB | 64.62 MB (-81.1%) |
| `bureau` | 1,716,428 | 17 | 472.82 MB | 85.12 MB | 387.70 MB (-82.0%) |
| `bureau_balance` | 27,299,925 | 3 | 1718.33 MB | 156.21 MB | 1562.11 MB (-90.9%) |
| `previous_application` | 1,670,214 | 37 | 1703.01 MB | 146.55 MB | 1556.46 MB (-91.4%) |
| `POS_CASH_balance` | 10,001,358 | 8 | 1060.95 MB | 209.84 MB | 851.11 MB (-80.2%) |
| `installments_payments` | 13,605,401 | 8 | 830.41 MB | 389.25 MB | 441.15 MB (-53.1%) |
| `credit_card_balance` | 3,840,312 | 23 | 846.39 MB | 292.99 MB | 553.40 MB (-65.4%) |
| **TOTAL** | **58,489,893** | - | **7216.58 MB** | **1390.37 MB** | **5826.21 MB (-80.7%)** |

### Categorical Columns (created during downcast)

- `application_train`: 16 categorical column(s) - `NAME_CONTRACT_TYPE`, `CODE_GENDER`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `NAME_TYPE_SUITE`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `OCCUPATION_TYPE`, `WEEKDAY_APPR_PROCESS_START`, `ORGANIZATION_TYPE`, `FONDKAPREMONT_MODE`, `HOUSETYPE_MODE`, `WALLSMATERIAL_MODE`, `EMERGENCYSTATE_MODE`
- `application_test`: 16 categorical column(s) - `NAME_CONTRACT_TYPE`, `CODE_GENDER`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `NAME_TYPE_SUITE`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `OCCUPATION_TYPE`, `WEEKDAY_APPR_PROCESS_START`, `ORGANIZATION_TYPE`, `FONDKAPREMONT_MODE`, `HOUSETYPE_MODE`, `WALLSMATERIAL_MODE`, `EMERGENCYSTATE_MODE`
- `bureau`: 3 categorical column(s) - `CREDIT_ACTIVE`, `CREDIT_CURRENCY`, `CREDIT_TYPE`
- `bureau_balance`: 1 categorical column(s) - `STATUS`
- `previous_application`: 16 categorical column(s) - `NAME_CONTRACT_TYPE`, `WEEKDAY_APPR_PROCESS_START`, `FLAG_LAST_APPL_PER_CONTRACT`, `NAME_CASH_LOAN_PURPOSE`, `NAME_CONTRACT_STATUS`, `NAME_PAYMENT_TYPE`, `CODE_REJECT_REASON`, `NAME_TYPE_SUITE`, `NAME_CLIENT_TYPE`, `NAME_GOODS_CATEGORY`, `NAME_PORTFOLIO`, `NAME_PRODUCT_TYPE`, `CHANNEL_TYPE`, `NAME_SELLER_INDUSTRY`, `NAME_YIELD_GROUP`, `PRODUCT_COMBINATION`
- `POS_CASH_balance`: 1 categorical column(s) - `NAME_CONTRACT_STATUS`
- `credit_card_balance`: 1 categorical column(s) - `NAME_CONTRACT_STATUS`

## 2. Main Table (`application_train`) Profiling

- **Shape**: `307,511` rows x `122` columns
- **Total Missing Cells**: `9,152,465` (24.4% of all cells)
- **Columns with Missing Values**: `67` / `122`

### Data Types Distribution
- `float32`: 65 columns
- `int8`: 37 columns
- `category`: 16 columns
- `int32`: 2 columns
- `int16`: 2 columns

### TARGET Class Balance
- **Class 0 (Repaid / Non-Default)**: 282,686 (91.93%)
- **Class 1 (Defaulted / Payment Difficulties)**: 24,825 (8.07%)
- **Class Imbalance Ratio**: ~ `11.39:1` (severe class imbalance)

### Top 15 Missing Columns
| Column | Missing Count | Missing % | Dtype |
| :--- | :--- | :--- | :--- |
| `COMMONAREA_MEDI` | 214,865 | 69.87% | `float32` |
| `COMMONAREA_MODE` | 214,865 | 69.87% | `float32` |
| `COMMONAREA_AVG` | 214,865 | 69.87% | `float32` |
| `NONLIVINGAPARTMENTS_MODE` | 213,514 | 69.43% | `float32` |
| `NONLIVINGAPARTMENTS_MEDI` | 213,514 | 69.43% | `float32` |
| `NONLIVINGAPARTMENTS_AVG` | 213,514 | 69.43% | `float32` |
| `FONDKAPREMONT_MODE` | 210,295 | 68.39% | `category` |
| `LIVINGAPARTMENTS_AVG` | 210,199 | 68.35% | `float32` |
| `LIVINGAPARTMENTS_MEDI` | 210,199 | 68.35% | `float32` |
| `LIVINGAPARTMENTS_MODE` | 210,199 | 68.35% | `float32` |
| `FLOORSMIN_MEDI` | 208,642 | 67.85% | `float32` |
| `FLOORSMIN_MODE` | 208,642 | 67.85% | `float32` |
| `FLOORSMIN_AVG` | 208,642 | 67.85% | `float32` |
| `YEARS_BUILD_MODE` | 204,488 | 66.50% | `float32` |
| `YEARS_BUILD_MEDI` | 204,488 | 66.50% | `float32` |

## 3. Join Keys & Cardinality Validation

Before merging, all primary and foreign key constraints are validated:

| Table | Join Key | Role | Status | Rows | Nulls (%) | Unique Keys | Cardinality Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `application_train` | `SK_ID_CURR` | Parent | PASS (Valid) | 307,511 | 0 (0.0%) | 307,511 | 1:1 (Unique) |
| `application_test` | `SK_ID_CURR` | Parent | PASS (Valid) | 48,744 | 0 (0.0%) | 48,744 | 1:1 (Unique) |
| `bureau` | `SK_ID_CURR` | Child | PASS (Valid) | 1,716,428 | 0 (0.0%) | 305,811 | 1:N (Avg 5.6 rows/key) |
| `previous_application` | `SK_ID_CURR` | Child | PASS (Valid) | 1,670,214 | 0 (0.0%) | 338,857 | 1:N (Avg 4.9 rows/key) |
| `POS_CASH_balance` | `SK_ID_CURR` | Child | PASS (Valid) | 10,001,358 | 0 (0.0%) | 337,252 | 1:N (Avg 29.7 rows/key) |
| `installments_payments` | `SK_ID_CURR` | Child | PASS (Valid) | 13,605,401 | 0 (0.0%) | 339,587 | 1:N (Avg 40.1 rows/key) |
| `credit_card_balance` | `SK_ID_CURR` | Child | PASS (Valid) | 3,840,312 | 0 (0.0%) | 103,558 | 1:N (Avg 37.1 rows/key) |
| `previous_application` | `SK_ID_PREV` | Parent | PASS (Valid) | 1,670,214 | 0 (0.0%) | 1,670,214 | 1:1 (Unique) |
| `POS_CASH_balance` | `SK_ID_PREV` | Child | PASS (Valid) | 10,001,358 | 0 (0.0%) | 936,325 | 1:N (Avg 10.7 rows/key) |
| `installments_payments` | `SK_ID_PREV` | Child | PASS (Valid) | 13,605,401 | 0 (0.0%) | 997,752 | 1:N (Avg 13.6 rows/key) |
| `credit_card_balance` | `SK_ID_PREV` | Child | PASS (Valid) | 3,840,312 | 0 (0.0%) | 104,307 | 1:N (Avg 36.8 rows/key) |
| `bureau` | `SK_ID_BUREAU` | Parent | PASS (Valid) | 1,716,428 | 0 (0.0%) | 1,716,428 | 1:1 (Unique) |
| `bureau_balance` | `SK_ID_BUREAU` | Child | PASS (Valid) | 27,299,925 | 0 (0.0%) | 817,395 | 1:N (Avg 33.4 rows/key) |

### Cross-Table Referential Integrity (Orphan Record Check)

| Join Key | Parent Table(s) | Child Table | Unique Keys in Child | Orphan Keys in Child | Integrity Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `SK_ID_CURR` | `application_train + test` | `bureau` | 305,811 | 0 (0.0%) | PASS (100% matched) |
| `SK_ID_CURR` | `application_train + test` | `previous_application` | 338,857 | 0 (0.0%) | PASS (100% matched) |
| `SK_ID_CURR` | `application_train + test` | `POS_CASH_balance` | 337,252 | 0 (0.0%) | PASS (100% matched) |
| `SK_ID_CURR` | `application_train + test` | `installments_payments` | 339,587 | 0 (0.0%) | PASS (100% matched) |
| `SK_ID_CURR` | `application_train + test` | `credit_card_balance` | 103,558 | 0 (0.0%) | PASS (100% matched) |
| `SK_ID_PREV` | `previous_application` | `POS_CASH_balance` | 936,325 | 37,422 (4.0%) | WARN (37,422 Orphans / 4.0%) |
| `SK_ID_PREV` | `previous_application` | `installments_payments` | 997,752 | 38,847 (3.89%) | WARN (38,847 Orphans / 3.89%) |
| `SK_ID_PREV` | `previous_application` | `credit_card_balance` | 104,307 | 11,372 (10.9%) | WARN (11,372 Orphans / 10.9%) |
| `SK_ID_BUREAU` | `bureau` | `bureau_balance` | 817,395 | 43,041 (5.27%) | WARN (43,041 Orphans / 5.27%) |

## 4. Warnings & Anomalies Detected

- [WARN] Table 'POS_CASH_balance' has 37,422 unique SK_ID_PREV values (4.0%) not found in previous_application
- [WARN] Table 'installments_payments' has 38,847 unique SK_ID_PREV values (3.89%) not found in previous_application
- [WARN] Table 'credit_card_balance' has 11,372 unique SK_ID_PREV values (10.9%) not found in previous_application
- [WARN] Table 'bureau_balance' has 43,041 unique SK_ID_BUREAU values (5.27%) not found in bureau

---

*Report generated by `server/data_loader.py` - Data validated. Ready for preprocessing and feature engineering.*