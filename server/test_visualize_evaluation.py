"""Unit tests for visualize_evaluation.py."""

import unittest
import sys
import tempfile
import json
from pathlib import Path

server_dir = Path(__file__).resolve().parent
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))

from visualize_evaluation import (
    load_evaluation_data,
    build_metrics_dataframe,
    plot_overall_benchmark,
    plot_cv_variance,
    plot_threshold_comparison,
    plot_confusion_matrices,
    plot_comprehensive_dashboard,
)


class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.sample_data = {
            "logistic_regression": {
                "auc_roc": 0.755,
                "pr_auc": 0.227,
                "ks_stat": 0.392,
                "optimal_threshold": 0.515,
                "default_threshold": {"f1": 0.27, "precision": 0.17, "recall": 0.68, "confusion_matrix": [[32000, 13000], [1200, 2700]]},
                "optimal_threshold_metrics": {"f1": 0.28, "precision": 0.18, "recall": 0.67, "confusion_matrix": [[33000, 12000], [1300, 2600]]},
                "cv_fold_aucs": [0.76, 0.75, 0.74, 0.75, 0.76],
                "cv_auc_mean": 0.752,
                "cv_auc_std": 0.007,
                "fit_time_sec": 14.6,
            },
            "stacking_ensemble": {
                "auc_roc": 0.769,
                "pr_auc": 0.248,
                "ks_stat": 0.402,
                "optimal_threshold": 0.062,
                "default_threshold": {"f1": 0.02, "precision": 0.35, "recall": 0.01, "confusion_matrix": [[45000, 50], [3900, 50]]},
                "optimal_threshold_metrics": {"f1": 0.26, "precision": 0.17, "recall": 0.72, "confusion_matrix": [[32000, 13000], [1100, 2900]]},
                "cv_fold_aucs": [0.77, 0.76, 0.77, 0.78, 0.76],
                "cv_auc_mean": 0.768,
                "cv_auc_std": 0.008,
                "fit_time_sec": 0.0,
            }
        }

    def test_dataframe_building(self):
        df = build_metrics_dataframe(self.sample_data)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.loc[0, "key"], "stacking_ensemble")  # Sorted by AUC

    def test_plots_generation(self):
        df = build_metrics_dataframe(self.sample_data)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            p1 = tmp_path / "benchmark.png"
            p2 = tmp_path / "cv.png"
            p3 = tmp_path / "thresh.png"
            p4 = tmp_path / "cm.png"
            p5 = tmp_path / "dash.png"

            plot_overall_benchmark(df, p1)
            plot_cv_variance(df, p2)
            plot_threshold_comparison(df, p3)
            plot_confusion_matrices(df, p4)
            plot_comprehensive_dashboard(df, p5)

            self.assertTrue(p1.exists() and p1.stat().st_size > 0)
            self.assertTrue(p2.exists() and p2.stat().st_size > 0)
            self.assertTrue(p3.exists() and p3.stat().st_size > 0)
            self.assertTrue(p4.exists() and p4.stat().st_size > 0)
            self.assertTrue(p5.exists() and p5.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
