"""Unit tests for train_models.py (verifying metrics, model instantiation, and PyTorch DNN)."""

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

server_dir = Path(__file__).resolve().parent
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))

from train_models import (
    compute_ks_statistic,
    evaluate_predictions,
    instantiate_estimator,
    PyTorchDNNWrapper,
    CreditDefaultDNN,
)


class TestMetrics(unittest.TestCase):
    def test_ks_statistic(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        ks_stat, opt_thresh = compute_ks_statistic(y_true, y_proba)
        self.assertAlmostEqual(ks_stat, 1.0)
        self.assertGreater(opt_thresh, 0.4)
        self.assertLessEqual(opt_thresh, 0.6)

    def test_evaluate_predictions(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        res = evaluate_predictions(y_true, y_proba)
        self.assertIn("auc_roc", res)
        self.assertIn("ks_stat", res)
        self.assertIn("pr_auc", res)
        self.assertEqual(res["auc_roc"], 1.0)


class TestDNN(unittest.TestCase):
    def test_pytorch_dnn_wrapper_fit_predict(self):
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=100)
        wrapper = PyTorchDNNWrapper(input_dim=10, hidden_dims=[16, 8], epochs=2, batch_size=32)
        wrapper.fit(X, y)
        probas = wrapper.predict_proba(X)
        self.assertEqual(probas.shape, (100, 2))
        self.assertTrue(np.all(probas >= 0.0) and np.all(probas <= 1.0))
        preds = wrapper.predict(X)
        self.assertEqual(len(preds), 100)


class TestEstimatorInstantiations(unittest.TestCase):
    def test_instantiate_all(self):
        for name in ["logistic_regression", "lda", "random_forest", "gradient_boosting", "xgboost", "lightgbm", "dnn"]:
            est = instantiate_estimator(
                name=name,
                best_params={},
                spw=11.4,
                class_weights={0: 0.54, 1: 6.19},
                input_dim=20,
                epochs=1,
            )
            self.assertIsNotNone(est)


if __name__ == "__main__":
    unittest.main()
