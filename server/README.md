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
    ├── fetch_data.py           # Kaggle downloader for the REAL dataset (with validation)
    ├── feature_engineering.py  # Satellite-table aggregation to SK_ID_CURR + application ratios
    ├── preprocessing.py        # ColumnTransformer (impute/encode/scale) + imbalance helpers
    ├── run_preprocessing.py    # Builds feature_table.parquet + fitted preprocessors
    ├── tune_hyperparameters.py # RandomizedSearchCV for 6 models -> best_hyperparameters.json
    ├── generate_sample_data.py # Synthetic sample generator for immediate testing
    ├── test_data_loader.py     # Automated unit tests
    └── requirements.txt        # Python dependencies
```

---

## ⚡ Key Features

1. **Safe Memory Downcasting (`reduce_mem_usage`)**:
   - Analyzes integer min/max boundaries and downcasts `int64` to `int8`, `int16`, `int32`.
   - Casts safe floating point numbers from `float64` to `float32`.
   - Converts object (string) columns to pandas `category` dtype by default - preserves
     values and NaN exactly while cutting string memory 60-90% on low-cardinality
     columns such as `NAME_CONTRACT_TYPE` or `CODE_GENDER` (opt out with
     `reduce_mem_usage(df, convert_categories=False)`).
   - Protects join identifiers (`SK_ID_CURR`, `SK_ID_PREV`, `SK_ID_BUREAU`) - IDs stay
     numeric, never categorical, preserving exact integer joins.
   - Typically reduces dataset RAM consumption by **30% - 65%**.

2. **Data Authenticity Guard (`check_data_authenticity`)**:
   - Fingerprint-checks every table against the known shape of the REAL Kaggle dataset
     (e.g. `application_train` must be at least 300,000 rows x 122 columns).
   - Flags synthetic samples (`server/generate_sample_data.py` output) or truncated
     downloads as `SYNTHETIC_OR_TRUNCATED` in the report (Section 0) and via CLI
     exit code 2 from `run_loader.py`.
   - **Preprocessing on synthetic data is blocked** until the real dataset is fetched.

3. **Main Table Profiling (`inspect_application_train`)**:
   - Table shape (rows x cols).
   - Dtype counts and breakdown.
   - Missing-value percentage (overall and per column sorted descending).
   - Target (`TARGET`) class balance, default rate, and class imbalance ratio (~11.5:1).

4. **Pre-Merge Join Key & Cardinality Validation (`validate_join_keys`)**:
   - Validates existence of join keys across all tables.
   - Checks for `NULL` / missing keys.
   - Determines cardinality: 1:1 (unique parent) vs 1:N (child tables).
   - **Cross-Table Referential Integrity**: Flags any orphan child records whose parent keys do not exist in `application_train + application_test`, `previous_application`, or `bureau`.

5. **Automated Quality Report Generation (`generate_data_quality_report`)**:
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

```bash
python server/run_loader.py --data-dir data --output-report data/data_quality_report.md
```

### 3. Run Preprocessing & Feature Engineering

Builds multi-table aggregations, fits the ColumnTransformer, and saves `data/feature_table.parquet` and the preprocessors:
```bash
python server/run_preprocessing.py --data-dir data --outdir models
```

### 4. Hyperparameter Tuning

Tunes 6 candidate models via 5-Fold Stratified CV:
```bash
python server/tune_hyperparameters.py --subsample 50000 --n-iter 25
```

### 5. Train & Evaluate Models (5-Fold Stratified CV)

Trains all traditional, ensemble, and AI-driven models, fits the Stacking Ensemble, and generates benchmark metrics:
```bash
python server/train_models.py --models all --subsample 50000 --epochs 10
```

### 6. Start the FastAPI Backend Server

Launches the REST API serving real-time predictions and model comparison metrics to the Next.js frontend:
```bash
uvicorn server.api:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Running Unit Tests

```bash
python -m unittest server/test_data_loader.py
python -m unittest server/test_preprocessing.py
python -m unittest server/test_tuning.py
python -m unittest server/test_train_models.py
python -m unittest server/test_api.py
```
