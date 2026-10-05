"""Home Credit Default Risk - Data Loading & Quality Validation Module.

This module provides a standalone, reusable data-loading pipeline for the
Home Credit Default Risk dataset (Kaggle). It handles:
- Memory-efficient loading with safe dtype downcasting
- Inspection of `application_train` (shape, dtypes, missingness, TARGET balance)
- Comprehensive join-key validation (SK_ID_CURR, SK_ID_PREV, SK_ID_BUREAU)
- Generation of detailed Data Quality Reports in Markdown & CLI formats
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Standard Home Credit Default Risk CSV filenames
DATA_FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "POS_CASH_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
}

# Ground-truth shape fingerprints of the REAL Kaggle Home Credit Default Risk tables.
# If a loaded table is smaller than these minimums (rows or columns), the files in
# data/ are almost certainly the bundled SYNTHETIC sample (server/generate_sample_data.py)
# or a truncated download, NOT the real competition data.
# Generated with a ~2% safety margin under the true row counts (e.g. train 307,511).
REAL_DATA_SHAPE_MINIMUMS = {
    "application_train": {"min_rows": 300_000, "min_cols": 122},
    "application_test": {"min_rows": 48_000, "min_cols": 121},
    "bureau": {"min_rows": 1_600_000, "min_cols": 17},
    "bureau_balance": {"min_rows": 27_000_000, "min_cols": 3},
    "previous_application": {"min_rows": 1_600_000, "min_cols": 37},
    "POS_CASH_balance": {"min_rows": 10_000_000, "min_cols": 8},
    "installments_payments": {"min_rows": 13_600_000, "min_cols": 8},
    "credit_card_balance": {"min_rows": 3_800_000, "min_cols": 8},
}

# Foreign key relationships: (Key, Parent Table, List of Child Tables)
JOIN_KEY_RELATIONSHIPS = [
    {
        "key": "SK_ID_CURR",
        "parents": ["application_train", "application_test"],
        "children": [
            "bureau",
            "previous_application",
            "POS_CASH_balance",
            "installments_payments",
            "credit_card_balance",
        ],
    },
    {
        "key": "SK_ID_PREV",
        "parents": ["previous_application"],
        "children": [
            "POS_CASH_balance",
            "installments_payments",
            "credit_card_balance",
        ],
    },
    {
        "key": "SK_ID_BUREAU",
        "parents": ["bureau"],
        "children": [
            "bureau_balance",
        ],
    },
]


def reduce_mem_usage(
    df: pd.DataFrame,
    verbose: bool = False,
    convert_categories: bool = True,
) -> Tuple[pd.DataFrame, float, float]:
    """Iterates through columns and safely downcasts numeric types to reduce memory.

    Parameters
    ----------
    df : pd.DataFrame
        Target dataframe to downcast.
    verbose : bool
        If True, log memory savings for individual columns.
    convert_categories : bool
        If True, convert object (string) columns to pandas ``category`` dtype.
        Category conversion preserves the exact string values (join/merge on
        category columns works like object columns), while typically cutting
        string-column memory by 60-90% on low-cardinality columns such as
        NAME_CONTRACT_TYPE or CODE_GENDER.

    Returns
    -------
    Tuple[pd.DataFrame, float, float]
        Tuple containing (modified dataframe, initial_memory_mb, final_memory_mb).

    Notes
    -----
    Join-key columns (SK_ID_CURR, SK_ID_PREV, SK_ID_BUREAU) are never
    categorical-converted, preserving exact integer joins.
    """
    start_mem = df.memory_usage(deep=True).sum() / (1024**2)

    for col in df.columns:
        col_type = df[col].dtype

        # Skip downcasting for join keys to prevent overflow or sign issues
        if col in ("SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"):
            if col_type == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if pd.api.types.is_numeric_dtype(df[col]):
                # Keep ID as int32 or int64 safely
                if df[col].isnull().any():
                    df[col] = df[col].astype("Int64")
                else:
                    df[col] = df[col].astype(np.int64 if df[col].max() > 2147483647 else np.int32)
            continue

        if pd.api.types.is_integer_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)

        elif pd.api.types.is_float_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            # Check if values fit safely in float32 without loss
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

        elif convert_categories and col_type == object:
            # pandas >= 1.5 casts object->category directly, preserving NaN
            # as missing values (no sentinel needed, no FutureWarning in pandas 2.x)
            df[col] = df[col].astype("category")
            if verbose:
                logger.info(f"  Column '{col}' -> category")

    end_mem = df.memory_usage(deep=True).sum() / (1024**2)
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        logger.info(
            f"Memory downcast: {start_mem:.2f} MB -> {end_mem:.2f} MB ({reduction:.1f}% reduction)"
        )

    return df, start_mem, end_mem


class HomeCreditDataLoader:
    """Standalone, reusable data-loading & validation pipeline for Home Credit Default Risk."""

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.tables: Dict[str, pd.DataFrame] = {}
        self.memory_stats: Dict[str, Dict[str, float]] = {}

    def get_table_path(self, table_name: str) -> Path:
        """Resolve file path for a dataset table."""
        filename = DATA_FILES.get(table_name, f"{table_name}.csv")
        return self.data_dir / filename

    def load_table(
        self,
        table_name: str,
        downcast: bool = True,
        nrows: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Loads a single CSV table, optionally downcasting types for memory efficiency.

        Parameters
        ----------
        table_name : str
            Name of table without .csv extension (e.g., 'application_train').
        downcast : bool
            Whether to run memory optimization.
        nrows : Optional[int]
            Limit rows loaded (useful for rapid testing / debugging).
        verbose : bool
            Log loading progress.

        Returns
        -------
        pd.DataFrame
            The loaded DataFrame.
        """
        path = self.get_table_path(table_name)
        if not path.is_file():
            raise FileNotFoundError(f"Table file not found: {path.resolve()}")

        if verbose:
            logger.info(f"Loading '{table_name}' from {path} (nrows={nrows or 'ALL'})...")

        df = pd.read_csv(path, nrows=nrows)
        initial_mem = df.memory_usage(deep=True).sum() / (1024**2)

        if downcast:
            df, start_mem, end_mem = reduce_mem_usage(df, verbose=False)
            self.memory_stats[table_name] = {
                "initial_mb": start_mem,
                "final_mb": end_mem,
                "savings_pct": 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0,
            }
            if verbose:
                logger.info(
                    f"Loaded '{table_name}': {df.shape[0]:,} rows, {df.shape[1]} cols | "
                    f"Memory: {start_mem:.2f} MB -> {end_mem:.2f} MB "
                    f"(-{self.memory_stats[table_name]['savings_pct']:.1f}%)"
                )
        else:
            self.memory_stats[table_name] = {
                "initial_mb": initial_mem,
                "final_mb": initial_mem,
                "savings_pct": 0.0,
            }
            if verbose:
                logger.info(
                    f"Loaded '{table_name}': {df.shape[0]:,} rows, {df.shape[1]} cols | "
                    f"Memory: {initial_mem:.2f} MB (downcast skipped)"
                )

        self.tables[table_name] = df
        return df

    def load_all(
        self,
        tables: Optional[List[str]] = None,
        downcast: bool = True,
        nrows: Optional[int] = None,
        ignore_missing: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Loads all specified CSV tables (defaults to all 8 standard tables)."""
        target_tables = tables or list(DATA_FILES.keys())
        loaded = {}

        for tname in target_tables:
            try:
                loaded[tname] = self.load_table(
                    tname, downcast=downcast, nrows=nrows, verbose=True
                )
            except FileNotFoundError as err:
                if ignore_missing:
                    logger.warning(f"Skipping table '{tname}': {err}")
                else:
                    raise

        return loaded

    def check_data_authenticity(
        self, tables: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """Detects whether loaded tables are the real Kaggle dataset or synthetic/truncated samples.

        Compares each table's shape against ground-truth minimums of the real
        Home Credit Default Risk dataset (REAL_DATA_SHAPE_MINIMUMS). A table
        that is too small in rows OR columns cannot be the real data.

        Returns a dict with:
            "is_authentic" : bool - True only if every loaded known table matches
            "verdict"      : 'AUTHENTIC' | 'SYNTHETIC_OR_TRUNCATED'
            "tables"       : per-table diagnostics (expected vs actual shape)
            "message"      : human-readable summary for reports/CLI
        """
        data = tables or self.tables
        table_details: Dict[str, Dict[str, Any]] = {}
        all_authentic = True

        for tname, df in data.items():
            minimums = REAL_DATA_SHAPE_MINIMUMS.get(tname)
            if minimums is None:
                continue  # unknown table: cannot fingerprint

            rows, cols = df.shape
            rows_ok = rows >= minimums["min_rows"]
            cols_ok = cols >= minimums["min_cols"]
            authentic = rows_ok and cols_ok
            all_authentic &= authentic

            reasons = []
            if not rows_ok:
                reasons.append(
                    f"rows {rows:,} < expected minimum {minimums['min_rows']:,}"
                )
            if not cols_ok:
                reasons.append(
                    f"columns {cols} < expected minimum {minimums['min_cols']}"
                )

            table_details[tname] = {
                "actual_shape": (rows, cols),
                "expected_min_shape": (
                    minimums["min_rows"],
                    minimums["min_cols"],
                ),
                "authentic": authentic,
                "reasons": reasons,
                "hint": (
                    "matches synthetic sample scale"
                    if not authentic
                    and rows <= 100_000
                    else ("" if authentic else "truncated download?")
                ),
            }

        verdict = "AUTHENTIC" if all_authentic else "SYNTHETIC_OR_TRUNCATED"
        if all_authentic:
            message = (
                "[OK] All loaded tables match the real Kaggle dataset scale. "
                "Safe to proceed with preprocessing."
            )
        else:
            synthetic_hint = (
                " These shapes match server/generate_sample_data.py output."
                if any(d["hint"] == "matches synthetic sample scale" for d in table_details.values())
                else ""
            )
            message = (
                "[WARN] Loaded data does NOT match the real Kaggle Home Credit "
                "dataset scale." + synthetic_hint + " Preprocessing/training on this "
                "data will NOT produce valid results. Run 'python server/fetch_data.py' "
                "to download the real dataset into data/."
            )

        return {
            "is_authentic": all_authentic,
            "verdict": verdict,
            "tables": table_details,
            "message": message,
        }

    @staticmethod
    def inspect_application_train(df: pd.DataFrame) -> Dict[str, Any]:
        """Inspects application_train.csv: shape, dtypes, missingness, and TARGET balance.

        Returns structured diagnostics for reports and validation.
        """
        shape = df.shape
        # Stringify dtypes BEFORE counting: each CategoricalDtype instance is a
        # distinct key in value_counts(), and str()-collapsing afterwards would
        # silently keep only the last one (all stringify to "category").
        dtypes_str = df.dtypes.apply(str).value_counts().to_dict()

        # Missing values
        total_cells = np.prod(shape)
        total_missing = df.isnull().sum().sum()
        overall_missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0

        missing_per_col = df.isnull().sum()
        missing_pct_per_col = (missing_per_col / len(df)) * 100
        missing_df = pd.DataFrame(
            {
                "missing_count": missing_per_col,
                "missing_pct": missing_pct_per_col,
                "dtype": df.dtypes.astype(str),
            }
        )
        missing_cols = (
            missing_df[missing_df["missing_count"] > 0]
            .sort_values(by="missing_pct", ascending=False)
        )

        # TARGET class balance
        target_stats: Dict[str, Any] = {}
        if "TARGET" in df.columns:
            counts = df["TARGET"].value_counts(dropna=False).to_dict()
            pcts = (df["TARGET"].value_counts(normalize=True, dropna=False) * 100).to_dict()
            c0 = counts.get(0, 0)
            c1 = counts.get(1, 0)
            imbalance_ratio = (c0 / c1) if c1 > 0 else float("inf")
            target_stats = {
                "counts": {str(k): int(v) for k, v in counts.items()},
                "percentages": {str(k): round(float(v), 2) for k, v in pcts.items()},
                "imbalance_ratio": round(imbalance_ratio, 2),
                "has_nulls": bool(df["TARGET"].isnull().any()),
            }
        else:
            target_stats = {"warning": "TARGET column not found in application_train"}

        return {
            "shape": shape,
            "dtypes_summary": dtypes_str,
            "total_cells": int(total_cells),
            "total_missing_cells": int(total_missing),
            "overall_missing_pct": round(float(overall_missing_pct), 2),
            "columns_with_missing_count": len(missing_cols),
            "missing_columns_detail": missing_cols,
            "target_stats": target_stats,
        }

    def validate_join_keys(
        self, tables: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """Validates join keys across all tables before any merging takes place.

        Performs:
        1. Key presence & null-value detection
        2. Cardinality & uniqueness checks (1:1 vs 1:N)
        3. Cross-table referential integrity (checking for orphan child records)
        """
        data = tables or self.tables
        validation_results: Dict[str, Any] = {
            "key_summary": [],
            "referential_integrity": [],
            "warnings": [],
        }

        # 1. Check each key's presence, nulls, and cardinality in each available table
        for rel in JOIN_KEY_RELATIONSHIPS:
            key = rel["key"]
            all_involved = rel["parents"] + rel["children"]

            for tname in all_involved:
                if tname not in data:
                    continue

                df = data[tname]
                has_key = key in df.columns

                if not has_key:
                    validation_results["warnings"].append(
                        f"Expected join key '{key}' is MISSING in table '{tname}'"
                    )
                    validation_results["key_summary"].append(
                        {
                            "table": tname,
                            "key": key,
                            "role": "Parent" if tname in rel["parents"] else "Child",
                            "status": "MISSING_KEY",
                            "rows": len(df),
                            "null_count": None,
                            "null_pct": None,
                            "nunique": None,
                            "is_unique": False,
                            "cardinality_type": "N/A",
                        }
                    )
                    continue

                null_count = int(df[key].isnull().sum())
                null_pct = round(null_count / len(df) * 100, 3) if len(df) > 0 else 0.0
                nunique = int(df[key].nunique(dropna=True))
                is_unique = (nunique == len(df)) and (null_count == 0)

                # Parent tables (e.g. application_train) are expected to be 1:1 (unique)
                role = "Parent" if tname in rel["parents"] else "Child"
                expected_unique = role == "Parent"

                cardinality_type = "1:1 (Unique)" if is_unique else f"1:N (Avg {len(df)/max(1, nunique):.1f} rows/key)"

                if null_count > 0:
                    validation_results["warnings"].append(
                        f"Join key '{key}' has {null_count:,} NULL values in table '{tname}' ({null_pct}%)"
                    )

                if expected_unique and not is_unique and null_count == 0:
                    validation_results["warnings"].append(
                        f"Parent table '{tname}' has duplicate '{key}' entries (not strictly 1:1)"
                    )

                validation_results["key_summary"].append(
                    {
                        "table": tname,
                        "key": key,
                        "role": role,
                        "status": "OK" if null_count == 0 else "CONTAINS_NULLS",
                        "rows": len(df),
                        "null_count": null_count,
                        "null_pct": null_pct,
                        "nunique": nunique,
                        "is_unique": is_unique,
                        "cardinality_type": cardinality_type,
                    }
                )

        # 2. Check cross-table referential integrity (orphan detection)
        # Check SK_ID_CURR: combined application_train + application_test as universal parent set
        if "application_train" in data or "application_test" in data:
            parent_ids = set()
            if "application_train" in data and "SK_ID_CURR" in data["application_train"].columns:
                parent_ids.update(data["application_train"]["SK_ID_CURR"].dropna().unique())
            if "application_test" in data and "SK_ID_CURR" in data["application_test"].columns:
                parent_ids.update(data["application_test"]["SK_ID_CURR"].dropna().unique())

            for child in ["bureau", "previous_application", "POS_CASH_balance", "installments_payments", "credit_card_balance"]:
                if child in data and "SK_ID_CURR" in data[child].columns:
                    child_ids = set(data[child]["SK_ID_CURR"].dropna().unique())
                    unmatched = child_ids - parent_ids
                    unmatched_count = len(unmatched)
                    unmatched_pct = round(unmatched_count / max(1, len(child_ids)) * 100, 2)

                    validation_results["referential_integrity"].append(
                        {
                            "key": "SK_ID_CURR",
                            "parent": "application_train + test",
                            "child": child,
                            "child_unique_ids": len(child_ids),
                            "unmatched_ids": unmatched_count,
                            "unmatched_pct": unmatched_pct,
                            "status": "VERIFIED" if unmatched_count == 0 else "ORPHANS_FOUND",
                        }
                    )
                    if unmatched_count > 0:
                        validation_results["warnings"].append(
                            f"Table '{child}' has {unmatched_count:,} unique SK_ID_CURR values ({unmatched_pct}%) not found in application files"
                        )

        # Check SK_ID_PREV: previous_application as parent
        if "previous_application" in data and "SK_ID_PREV" in data["previous_application"].columns:
            prev_ids = set(data["previous_application"]["SK_ID_PREV"].dropna().unique())
            for child in ["POS_CASH_balance", "installments_payments", "credit_card_balance"]:
                if child in data and "SK_ID_PREV" in data[child].columns:
                    child_ids = set(data[child]["SK_ID_PREV"].dropna().unique())
                    unmatched = child_ids - prev_ids
                    unmatched_count = len(unmatched)
                    unmatched_pct = round(unmatched_count / max(1, len(child_ids)) * 100, 2)

                    validation_results["referential_integrity"].append(
                        {
                            "key": "SK_ID_PREV",
                            "parent": "previous_application",
                            "child": child,
                            "child_unique_ids": len(child_ids),
                            "unmatched_ids": unmatched_count,
                            "unmatched_pct": unmatched_pct,
                            "status": "VERIFIED" if unmatched_count == 0 else "ORPHANS_FOUND",
                        }
                    )
                    if unmatched_count > 0:
                        validation_results["warnings"].append(
                            f"Table '{child}' has {unmatched_count:,} unique SK_ID_PREV values ({unmatched_pct}%) not found in previous_application"
                        )

        # Check SK_ID_BUREAU: bureau as parent for bureau_balance
        if "bureau" in data and "bureau_balance" in data:
            if "SK_ID_BUREAU" in data["bureau"].columns and "SK_ID_BUREAU" in data["bureau_balance"].columns:
                bureau_ids = set(data["bureau"]["SK_ID_BUREAU"].dropna().unique())
                child_ids = set(data["bureau_balance"]["SK_ID_BUREAU"].dropna().unique())
                unmatched = child_ids - bureau_ids
                unmatched_count = len(unmatched)
                unmatched_pct = round(unmatched_count / max(1, len(child_ids)) * 100, 2)

                validation_results["referential_integrity"].append(
                    {
                        "key": "SK_ID_BUREAU",
                        "parent": "bureau",
                        "child": "bureau_balance",
                        "child_unique_ids": len(child_ids),
                        "unmatched_ids": unmatched_count,
                        "unmatched_pct": unmatched_pct,
                        "status": "VERIFIED" if unmatched_count == 0 else "ORPHANS_FOUND",
                    }
                )
                if unmatched_count > 0:
                    validation_results["warnings"].append(
                        f"Table 'bureau_balance' has {unmatched_count:,} unique SK_ID_BUREAU values ({unmatched_pct}%) not found in bureau"
                    )

        return validation_results

    def generate_data_quality_report(
        self,
        output_path: Optional[str | Path] = None,
        top_n_missing: int = 15,
    ) -> str:
        """Generates a comprehensive Markdown report summarizing:
        - Row counts, column counts, memory usage & downcasting savings
        - application_train shape, dtypes, missingness, and target distribution
        - Join key validation, cardinality, and referential integrity
        - Identified anomalies / warnings
        """
        lines = []
        lines.append("# Home Credit Default Risk - Data Quality & Loading Report\n")
        lines.append(f"> **Generated at**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"> **Data Directory**: `{self.data_dir.resolve()}`\n")

        # 0. Data Authenticity Check (real Kaggle data vs synthetic/truncated sample)
        authenticity = self.check_data_authenticity(self.tables)
        lines.append("## 0. Data Authenticity Check\n")
        if authenticity["is_authentic"]:
            lines.append(f"- **Verdict**: `{authenticity['verdict']}` - {authenticity['message']}\n")
        else:
            lines.append(f"- **Verdict**: `:x: {authenticity['verdict']}`")
            lines.append(f"- {authenticity['message']}\n")
            lines.append("| Table | Actual Shape | Expected Minimum (rows, cols) | Status |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for tname, detail in authenticity["tables"].items():
                rows, cols = detail["actual_shape"]
                erows, ecols = detail["expected_min_shape"]
                status = "OK" if detail["authentic"] else "TOO SMALL (synthetic/truncated)"
                lines.append(
                    f"| `{tname}` | {rows:,} x {cols} | {erows:,} x {ecols} | {status} |"
                )
            lines.append("")

        # 1. Overview Table of Loaded Files
        lines.append("## 1. Dataset Overview & Memory Footprint\n")
        lines.append(
            "| Table | Rows | Columns | Initial Memory | Downcast Memory | Memory Saved |"
        )
        lines.append(
            "| :--- | :--- | :--- | :--- | :--- | :--- |"
        )

        total_init_mem = 0.0
        total_final_mem = 0.0
        total_rows = 0

        for tname, df in self.tables.items():
            stats = self.memory_stats.get(
                tname,
                {"initial_mb": 0.0, "final_mb": 0.0, "savings_pct": 0.0},
            )
            init_m = stats["initial_mb"]
            final_m = stats["final_mb"]
            saved_pct = stats["savings_pct"]

            total_init_mem += init_m
            total_final_mem += final_m
            total_rows += len(df)
            saved_mb = max(0.0, init_m - final_m)

            lines.append(
                f"| `{tname}` | {len(df):,} | {df.shape[1]} | {init_m:.2f} MB | {final_m:.2f} MB | {saved_mb:.2f} MB (-{saved_pct:.1f}%) |"
            )

        total_savings = (
            100 * (total_init_mem - total_final_mem) / total_init_mem
            if total_init_mem > 0
            else 0.0
        )
        lines.append(
            f"| **TOTAL** | **{total_rows:,}** | - | **{total_init_mem:.2f} MB** | **{total_final_mem:.2f} MB** | **{max(0.0, total_init_mem - total_final_mem):.2f} MB (-{total_savings:.1f}%)** |\n"
        )

        # 1b. Categorical columns created by downcasting
        cat_lines = []
        for tname, df in self.tables.items():
            cat_cols = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
            if cat_cols:
                cat_lines.append(
                    f"- `{tname}`: {len(cat_cols)} categorical column(s) - {', '.join(f'`{c}`' for c in cat_cols)}"
                )
        if cat_lines:
            lines.append("### Categorical Columns (created during downcast)\n")
            lines.extend(cat_lines)
            lines.append("")

        # 2. Detailed Inspection of application_train
        if "application_train" in self.tables:
            train_df = self.tables["application_train"]
            app_stats = self.inspect_application_train(train_df)

            lines.append("## 2. Main Table (`application_train`) Profiling\n")
            lines.append(f"- **Shape**: `{app_stats['shape'][0]:,}` rows x `{app_stats['shape'][1]}` columns")
            lines.append(f"- **Total Missing Cells**: `{app_stats['total_missing_cells']:,}` ({app_stats['overall_missing_pct']}% of all cells)")
            lines.append(f"- **Columns with Missing Values**: `{app_stats['columns_with_missing_count']}` / `{app_stats['shape'][1]}`")
            
            # Dtypes summary
            lines.append("\n### Data Types Distribution")
            for dtype_name, count in app_stats["dtypes_summary"].items():
                lines.append(f"- `{dtype_name}`: {count} columns")

            # TARGET Balance
            target = app_stats["target_stats"]
            lines.append("\n### TARGET Class Balance")
            if "counts" in target:
                c0 = target["counts"].get("0", 0)
                c1 = target["counts"].get("1", 0)
                p0 = target["percentages"].get("0", 0)
                p1 = target["percentages"].get("1", 0)
                lines.append(f"- **Class 0 (Repaid / Non-Default)**: {c0:,} ({p0}%)")
                lines.append(f"- **Class 1 (Defaulted / Payment Difficulties)**: {c1:,} ({p1}%)")
                lines.append(f"- **Class Imbalance Ratio**: ~ `{target['imbalance_ratio']}:1` (severe class imbalance)")
                if target["has_nulls"]:
                    lines.append("- [WARN]: TARGET column contains missing values!")
            else:
                lines.append(f"- [WARN] {target.get('warning', 'TARGET stats unavailable')}")

            # Top missing columns
            missing_detail = app_stats["missing_columns_detail"]
            if not missing_detail.empty:
                lines.append(f"\n### Top {min(top_n_missing, len(missing_detail))} Missing Columns")
                lines.append("| Column | Missing Count | Missing % | Dtype |")
                lines.append("| :--- | :--- | :--- | :--- |")
                for col_name, row in missing_detail.head(top_n_missing).iterrows():
                    lines.append(
                        f"| `{col_name}` | {int(row['missing_count']):,} | {row['missing_pct']:.2f}% | `{row['dtype']}` |"
                    )
            lines.append("")

        # 3. Join Keys & Cardinality Validation
        validation = self.validate_join_keys(self.tables)
        lines.append("## 3. Join Keys & Cardinality Validation\n")
        lines.append("Before merging, all primary and foreign key constraints are validated:\n")
        lines.append("| Table | Join Key | Role | Status | Rows | Nulls (%) | Unique Keys | Cardinality Type |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for item in validation["key_summary"]:
            status_badge = "PASS (Valid)" if item["status"] == "OK" else f"WARN ({item['status']})"
            lines.append(
                f"| `{item['table']}` | `{item['key']}` | {item['role']} | {status_badge} | "
                f"{item['rows']:,} | {item['null_count']:,} ({item['null_pct']}%) | "
                f"{item['nunique']:,} | {item['cardinality_type']} |"
            )
        lines.append("")

        # 4. Cross-Table Referential Integrity
        lines.append("### Cross-Table Referential Integrity (Orphan Record Check)\n")
        lines.append("| Join Key | Parent Table(s) | Child Table | Unique Keys in Child | Orphan Keys in Child | Integrity Status |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

        for ref in validation["referential_integrity"]:
            status_badge = "PASS (100% matched)" if ref["unmatched_ids"] == 0 else f"WARN ({ref['unmatched_ids']:,} Orphans / {ref['unmatched_pct']}%)"
            lines.append(
                f"| `{ref['key']}` | `{ref['parent']}` | `{ref['child']}` | "
                f"{ref['child_unique_ids']:,} | {ref['unmatched_ids']:,} ({ref['unmatched_pct']}%) | {status_badge} |"
            )
        lines.append("")

        # 5. Anomalies and Warnings
        lines.append("## 4. Warnings & Anomalies Detected\n")
        if not authenticity["is_authentic"]:
            # The message already carries its own [WARN] prefix
            lines.append(f"- [CRITICAL] {authenticity['message'].replace('[WARN] ', '')}")
        if validation["warnings"]:
            for warn in validation["warnings"]:
                lines.append(f"- [WARN] {warn}")
        else:
            lines.append("- [PASS] **No critical anomalies or key mismatches detected.** All foreign keys cleanly link to parent tables.")

        lines.append("\n---\n")
        if authenticity["is_authentic"]:
            lines.append("*Report generated by `server/data_loader.py` - Data validated. Ready for preprocessing and feature engineering.*")
        else:
            lines.append("*Report generated by `server/data_loader.py` - :warning: Data NOT ready for preprocessing: replace synthetic/truncated files with the real Kaggle dataset (`python server/fetch_data.py`).*")

        report_content = "\n".join(lines)

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(report_content, encoding="utf-8")
            logger.info(f"Data quality report saved to {out_file.resolve()}")

        return report_content
