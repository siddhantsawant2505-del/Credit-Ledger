"""Fast unit tests for tune_hyperparameters.py (no full-data searches here)."""

import unittest
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

server_dir = Path(__file__).resolve().parent
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))

from tune_hyperparameters import (
    RANDOM_STATE,
    build_model_specs,
    load_tuning_data,
    run_search,
    save_results,
    _jsonable,
)


class TestSpecs(unittest.TestCase):
    def test_all_six_models_present(self):
        specs = build_model_specs(spw=11.4, class_weights={0: 0.54, 1: 6.19})
        expected = {
            "logistic_regression", "lda", "random_forest",
            "gradient_boosting", "xgboost", "lightgbm",
        }
        self.assertEqual(set(specs.keys()), expected)

    def test_required_hyperparameters_in_spaces(self):
        specs = build_model_specs(11.4, {0: 0.54, 1: 6.19})
        self.assertIn("C", specs["logistic_regression"]["params"])
        self.assertIn("penalty", specs["logistic_regression"]["params"])
        self.assertIn("max_depth", specs["random_forest"]["params"])
        self.assertIn("max_features", specs["random_forest"]["params"])
        for m in ("xgboost", "lightgbm"):
            for p in ("n_estimators", "learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
                self.assertIn(p, specs[m]["params"], f"{m} missing {p}")
        self.assertIn("subsample", specs["gradient_boosting"]["params"])

    def test_n_iter_within_budget(self):
        specs = build_model_specs(11.4, {0: 0.54, 1: 6.19})
        for name, spec in specs.items():
            self.assertLessEqual(spec["n_iter"], 30, f"{name} exceeds budget")


class TestRunner(unittest.TestCase):
    def _tiny_data(self, n=400):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            {
                "f1": rng.normal(size=n),
                "f2": rng.normal(size=n),
                "f3": rng.integers(0, 4, n).astype(float),
            }
        )
        # f1 genuinely predictive
        logits = 2.0 * X["f1"] + 0.1 * rng.normal(size=n)
        y = pd.Series((logits > logits.quantile(0.92)).astype(np.int8), name="TARGET")
        return X, y

    def test_run_search_logistic(self):
        X, y = self._tiny_data()
        specs = build_model_specs(11.4, {0: 0.54, 1: 6.19})
        res = run_search(
            "logistic_regression", specs["logistic_regression"], X.values, y, spw=11.4,
            n_iter_override=3,
        )
        self.assertIn("best_cv_auc", res)
        self.assertGreater(res["best_cv_auc"], 0.5)  # f1 is predictive
        self.assertIn("C", res["best_params"])
        self.assertEqual(res["random_state"], RANDOM_STATE)
        self.assertLessEqual(len(res["top5"]), 5)
        # JSON serializable
        json.dumps(res, default=_jsonable)

    def test_run_search_lightgbm(self):
        X, y = self._tiny_data()
        res = run_search(
            "lightgbm",
            build_model_specs(11.4, {0: 0.54, 1: 6.19})["lightgbm"],
            X.values, y, spw=11.4, n_iter_override=2,
        )
        self.assertGreater(res["best_cv_auc"], 0.5)
        json.dumps(res, default=_jsonable)

    def test_subsample_stratified(self):
        n = 1000
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "f1": rng.normal(size=n),
                "TARGET": rng.choice([0, 1], size=n, p=[0.92, 0.08]),
                "SPLIT": "train",
            }
        )
        X, y = load_tuning_data.__wrapped__(df_path=None, subsample=400, full=False) if False else (None, None)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ft.parquet"
            df.to_parquet(p, index=False)
            X, y = load_tuning_data(p, subsample=400, full=False)
        self.assertEqual(len(y), 400)
        # stratification preserved within ~1pp
        self.assertLess(abs(y.mean() - 0.08), 0.012)


class TestPersistence(unittest.TestCase):
    def test_save_results_roundtrip(self):
        results = {
            "meta": {"data_size": 100, "generated_at": "t", "random_state": 42,
                      "cv_folds": 5, "scoring": "roc_auc"},
            "imbalance": {"scale_pos_weight": 11.4, "class_weight": {0: 0.5, 1: 6.0}},
            "models": {
                "lightgbm": {
                    "model": "lightgbm", "best_cv_auc": 0.75,
                    "best_params": {"n_estimators": 500, "learning_rate": 0.05},
                    "n_iter": 25, "n_candidates": 25, "elapsed_sec": 1.0,
                    "random_state": 42, "cv_folds": 5, "scoring": "roc_auc",
                    "top5": [{"params": {}, "mean_auc": 0.75}],
                    "data_size": 100,
                }
            },
        }
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            jp, mp = Path(tmp) / "r.json", Path(tmp) / "r.md"
            save_results(results, jp, mp)
            loaded = json.loads(jp.read_text(encoding="utf-8"))
            self.assertEqual(loaded["models"]["lightgbm"]["best_params"]["n_estimators"], 500)
            md = mp.read_text(encoding="utf-8")
            self.assertIn("lightgbm", md)
            self.assertIn("0.7500", md)


if __name__ == "__main__":
    unittest.main()
