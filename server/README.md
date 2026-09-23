# Home Credit Default Risk - Backend Server & Data Loader

This module provides an optimized, production-grade data-loading, dtype-downcasting, and join-key validation pipeline for the **Home Credit Default Risk** Kaggle dataset.

---

## 📁 Directory Architecture

```text
Credit Ledger/
├── client/                     # Frontend application directory
├── data/                       # Dataset directory (place Kaggle CSVs here)
│   ├── application_train.csv
│   ├── application_test.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── previous_application.csv
│   ├── POS_CASH_balance.csv
│   ├── installments_payments.csv
│   ├── credit_card_balance.csv
│   └── data_quality_report.md  # Generated quality report
└── server/                     # Backend Python modules
    ├── data_loader.py          # Core reusable loading, downcasting & validation engine
    ├── run_loader.py           # Command-line entrypoint
    ├── generate_sample_data.py # Synthetic sample generator for immediate testing
    ├── test_data_loader.py     # Automated unit tests
    └── requirements.txt        # Python dependencies
```

---

## ⚡ Key Features

1. **Safe Memory Downcasting (`reduce_mem_usage`)**:
   - Analyzes integer min/max boundaries and downcasts `int64` to `int8`, `int16`, `int32`.
   - Casts safe floating point numbers from `float64` to `float32`.
   - Protects join identifiers (`SK_ID_CURR`, `SK_ID_PREV`, `SK_ID_BUREAU`) from improper downcasting.
   - Typically reduces dataset RAM consumption by **30% - 65%**.

2. **Main Table Profiling (`inspect_application_train`)**:
   - Table shape (rows x cols).
   - Dtype counts and breakdown.
   - Missing-value percentage (overall and per column sorted descending).
   - Target (`TARGET`) class balance, default rate, and class imbalance ratio (~11.5:1).

3. **Pre-Merge Join Key & Cardinality Validation (`validate_join_keys`)**:
   - Validates existence of join keys across all tables.
   - Checks for `NULL` / missing keys.
   - Determines cardinality: 1:1 (unique parent) vs 1:N (child tables).
   - **Cross-Table Referential Integrity**: Flags any orphan child records whose parent keys do not exist in `application_train + application_test`, `previous_application`, or `bureau`.

4. **Automated Quality Report Generation (`generate_data_quality_report`)**:
   - Outputs clean Markdown tables summarizing row counts, memory before/after, missingness, cardinality, and referential integrity badges.
   - Automatically writes to file (e.g. `data/data_quality_report.md`) or outputs to stdout.

---

## 🚀 Getting Started

### 1. Installation

Ensure you have Python 3.10+ installed:
```bash
pip install -r server/requirements.txt
```

### 2. Run the Data Loader & Validator CLI

To load the dataset from `data/` and generate a report:
```bash
python server/run_loader.py --data-dir data --output-report data/data_quality_report.md
```

#### Optional CLI Flags:
- `--nrows 10000`: Load only first N rows for rapid testing.
- `--no-downcast`: Disable memory optimization.
- `--tables application_train bureau`: Load specific tables only.

### 3. Programmatic Usage in Python

```python
from server.data_loader import HomeCreditDataLoader

# Initialize loader pointing to data directory
loader = HomeCreditDataLoader(data_dir="data")

# 1. Load single table with automatic dtype downcasting
train_df = loader.load_table("application_train", downcast=True)

# 2. Inspect application_train
stats = loader.inspect_application_train(train_df)
print(f"Shape: {stats['shape']}")
print(f"Target distribution: {stats['target_stats']}")

# 3. Load all tables and validate join relationships
tables = loader.load_all(downcast=True)
validation = loader.validate_join_keys(tables)

# 4. Generate and save Data Quality Report
report_md = loader.generate_data_quality_report(output_path="data/data_quality_report.md")
```

### 4. Running Unit Tests

```bash
python -m unittest server/test_data_loader.py
```
