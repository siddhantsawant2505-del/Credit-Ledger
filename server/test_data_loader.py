"""Unit tests for the Home Credit Data Loader & Validation module."""

import unittest
import sys
from pathlib import Path

# Add server directory to path
server_dir = Path(__file__).resolve().parent
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))

import numpy as np
import pandas as pd
from data_loader import (
    REAL_DATA_SHAPE_MINIMUMS,
    reduce_mem_usage,
    HomeCreditDataLoader,
)


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create minimal synthetic DataFrames
        self.df_train = pd.DataFrame(
            {
                "SK_ID_CURR": [1001, 1002, 1003, 1004],
                "TARGET": [0, 0, 1, 0],
                "AMT_INCOME": [100000.0, 150000.0, 80000.0, 120000.0],
                "FLAG_CAR": [1, 0, 1, 0],
                "NULL_COL": [np.nan, 1.0, np.nan, 2.0],
            }
        )

        self.df_bureau = pd.DataFrame(
            {
                "SK_ID_CURR": [1001, 1001, 1002, 9999],  # 9999 is orphan ID
                "SK_ID_BUREAU": [501, 502, 503, 504],
                "AMT_CREDIT": [5000.0, 10000.0, 20000.0, 15000.0],
            }
        )

        self.tables = {
            "application_train": self.df_train,
            "bureau": self.df_bureau,
        }

    def test_memory_reduction(self):
        df = pd.DataFrame(
            {
                "small_int": np.array([1, 2, 3], dtype=np.int64),
                "med_int": np.array([1000, 2000, 3000], dtype=np.int64),
                "float_val": np.array([1.5, 2.5, 3.5], dtype=np.float64),
            }
        )
        downcasted, start_mem, end_mem = reduce_mem_usage(df)
        self.assertLess(end_mem, start_mem)
        self.assertEqual(downcasted["small_int"].dtype, np.int8)
        self.assertEqual(downcasted["med_int"].dtype, np.int16)
        self.assertEqual(downcasted["float_val"].dtype, np.float32)

    def test_categorical_downcast(self):
        """Object columns convert to category dtype by default, preserving values."""
        # Use a realistic frame size: category conversion pays off at scale,
        # not on toy 4-row frames where overhead cancels the savings.
        n = 1_000
        df = pd.DataFrame(
            {
                "contract": np.random.default_rng(0).choice(["Cash loans", "Revolving loans"], size=n),
                "gender": np.random.default_rng(1).choice(["M", "F"], size=n),
                "id": np.arange(n),
            }
        )
        original_mem = df.memory_usage(deep=True).sum() / (1024**2)
        downcasted, _, end_mem = reduce_mem_usage(df)

        self.assertIsInstance(downcasted["contract"].dtype, pd.CategoricalDtype)
        self.assertIsInstance(downcasted["gender"].dtype, pd.CategoricalDtype)

        # Values preserved exactly (order + content)
        self.assertEqual(list(downcasted["contract"]), list(df["contract"]))
        self.assertEqual(list(downcasted["gender"]), list(df["gender"]))

        # Categorical conversion must actually save memory on string-heavy frames
        self.assertLess(end_mem, original_mem)

    def test_categorical_nan_preserved(self):
        """NaN values survive object -> category conversion (no sentinel leakage)."""
        df = pd.DataFrame({"with_nan": ["A", None, "B", "A"]})
        downcasted, _, _ = reduce_mem_usage(df)
        self.assertIsInstance(downcasted["with_nan"].dtype, pd.CategoricalDtype)
        self.assertTrue(downcasted["with_nan"].isnull().any())
        # NaN must NOT have become a category of its own
        self.assertNotIn("__MISSING__", list(downcasted["with_nan"].cat.categories))

    def test_opt_out_categorical(self):
        """convert_categories=False keeps object columns untouched."""
        df = pd.DataFrame({"contract": ["Cash", "Revolving"]})
        downcasted, _, _ = reduce_mem_usage(df, convert_categories=False)
        self.assertEqual(downcasted["contract"].dtype, object)

    def test_categorical_join_keys_untouched(self):
        """ID columns stay numeric even with categorical conversion on."""
        df = pd.DataFrame(
            {
                "SK_ID_CURR": [100001, 100002, 100003],
                "contract": ["Cash", "Cash", "Revolving"],
            }
        )
        downcasted, _, _ = reduce_mem_usage(df)
        self.assertTrue(pd.api.types.is_integer_dtype(downcasted["SK_ID_CURR"]))
        self.assertIsInstance(downcasted["contract"].dtype, pd.CategoricalDtype)

    def test_inspect_application_train(self):
        stats = HomeCreditDataLoader.inspect_application_train(self.df_train)
        self.assertEqual(stats["shape"], (4, 5))
        self.assertEqual(stats["target_stats"]["counts"]["0"], 3)
        self.assertEqual(stats["target_stats"]["counts"]["1"], 1)
        self.assertEqual(stats["target_stats"]["percentages"]["0"], 75.0)
        self.assertEqual(stats["target_stats"]["percentages"]["1"], 25.0)
        self.assertEqual(stats["target_stats"]["imbalance_ratio"], 3.0)
        self.assertIn("NULL_COL", stats["missing_columns_detail"].index)

    def test_validate_join_keys_orphan_detection(self):
        loader = HomeCreditDataLoader()
        validation = loader.validate_join_keys(self.tables)

        # Ensure orphan ID 9999 is flagged
        ref_checks = validation["referential_integrity"]
        bureau_check = next(r for r in ref_checks if r["child"] == "bureau")
        self.assertEqual(bureau_check["unmatched_ids"], 1)
        self.assertEqual(bureau_check["status"], "ORPHANS_FOUND")

    def test_generate_report(self):
        loader = HomeCreditDataLoader()
        loader.tables = self.tables
        loader.memory_stats = {
            "application_train": {"initial_mb": 0.05, "final_mb": 0.02, "savings_pct": 60.0},
            "bureau": {"initial_mb": 0.04, "final_mb": 0.01, "savings_pct": 75.0},
        }
        report = loader.generate_data_quality_report()
        self.assertIn("# Home Credit Default Risk - Data Quality & Loading Report", report)
        self.assertIn("application_train", report)
        self.assertIn("bureau", report)
        self.assertIn("TARGET Class Balance", report)
        # Authenticity section must always be present
        self.assertIn("Data Authenticity Check", report)

    def test_authenticity_rejects_synthetic(self):
        """Tiny tables (like the bundled 10k-row sample) must be flagged."""
        loader = HomeCreditDataLoader()
        result = loader.check_data_authenticity(self.tables)
        self.assertFalse(result["is_authentic"])
        self.assertEqual(result["verdict"], "SYNTHETIC_OR_TRUNCATED")
        self.assertFalse(result["tables"]["application_train"]["authentic"])
        self.assertIn("generate_sample_data", result["message"])

    def test_authenticity_accepts_real_scale(self):
        """DataFrames shaped like the real dataset (with correct IDs) pass."""
        n = REAL_DATA_SHAPE_MINIMUMS["application_train"]["min_rows"]
        n_cols = REAL_DATA_SHAPE_MINIMUMS["application_train"]["min_cols"]
        real_scale_train = pd.DataFrame(
            {
                "SK_ID_CURR": np.arange(100001, 100001 + n),
                **{f"feat_{i}": np.zeros(n) for i in range(n_cols - 1)},
            }
        )
        loader = HomeCreditDataLoader()
        result = loader.check_data_authenticity({"application_train": real_scale_train})
        self.assertTrue(result["is_authentic"])
        self.assertEqual(result["verdict"], "AUTHENTIC")

    def test_minimums_cover_all_standard_tables(self):
        """Every standard table has a fingerprint (guards against typos)."""
        from data_loader import DATA_FILES

        self.assertEqual(set(REAL_DATA_SHAPE_MINIMUMS.keys()), set(DATA_FILES.keys()))


if __name__ == "__main__":
    unittest.main()
