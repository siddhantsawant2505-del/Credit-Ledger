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
from data_loader import reduce_mem_usage, HomeCreditDataLoader


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


if __name__ == "__main__":
    unittest.main()
