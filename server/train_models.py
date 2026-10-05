"""Home Credit Default Risk - Model Training & Evaluation Module.

Trains traditional, ensemble, stacking, and AI-driven models using the best
hyperparameters tuned in `tune_hyperparameters.py`. Performs 5-fold stratified
cross-validation to produce Out-Of-Fold (OOF) metrics, fits final models on full
training data, and persists artifacts in `models/`.

Models:
1. Traditional Baselines: Logistic Regression, Linear Discriminant Analysis (LDA)
2. Ensemble ML: Random Forest, Gradient Boosting, XGBoost, LightGBM, Stacking Ensemble
3. AI-Driven: Deep Neural Network (DNN) in PyTorch

Metrics evaluated:
- ROC-AUC
- PR-AUC (Average Precision)
- F1-Score (at default 0.5 and optimal threshold)
- Precision & Recall
- Kolmogorov-Smirnov (KS) statistic
- Confusion Matrix

Usage:
    python server/train_models.py [--models all] [--subsample 0] [--epochs 15]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from feature_engineering import TARGET
from preprocessing import ID_CURR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
CV_FOLDS = 5

# Chunked training / resume support: per-model OOF predictions and fold state
# are persisted under models/oof_cache/ so a long full-data run can be split
# across multiple process launches (each fold is saved as it completes).
OOF_CACHE_DIRNAME = "oof_cache"
OOF_CACHE_VERSION = 1


def _data_key(X: np.ndarray) -> str:
    """Stable identity for the training matrix (rows x cols) for cache checks."""
    return f"{X.shape[0]}x{X.shape[1]}"


def transform_f32(preprocessor: Any, X_df: pd.DataFrame, chunk_rows: int = 100_000) -> np.ndarray:
    """Transform in row-chunks, casting each to float32 immediately.

    Preprocessors here are row-wise at transform time (imputer/scaler/encoders
    hold no cross-row state), so chunking is numerically identical to a single
    transform but caps the float64 intermediate at ~190 MiB instead of ~600 MiB
    on 307k rows - matters on RAM-constrained machines.
    """
    parts: List[np.ndarray] = []
    for i in range(0, len(X_df), chunk_rows):
        t = preprocessor.transform(X_df.iloc[i : i + chunk_rows])
        if hasattr(t, "toarray"):
            t = t.toarray()
        parts.append(np.asarray(t, dtype=np.float32))
    return np.vstack(parts)


def _fold_state_path(models_dir: Path, name: str) -> Path:
    return models_dir / OOF_CACHE_DIRNAME / f"{name}.npz"


def save_oof_cache(
    models_dir: Path,
    name: str,
    oof_proba: np.ndarray,
    fold_aucs: List[float],
    next_fold: int,
    data_key: str,
) -> None:
    """Persist OOF predictions + fold progress for one model (crash-safe resume)."""
    d = models_dir / OOF_CACHE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _fold_state_path(models_dir, name),
        oof_proba=oof_proba,
        fold_aucs=np.asarray(fold_aucs, dtype=np.float64),
        next_fold=np.int64(next_fold),
        data_key=np.array(data_key),
        version=np.int64(OOF_CACHE_VERSION),
    )


def load_oof_cache(models_dir: Path, name: str, data_key: str) -> Optional[Dict[str, Any]]:
    """Load cached OOF state for a model, or None if absent/incompatible."""
    p = _fold_state_path(models_dir, name)
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            if int(z["version"]) != OOF_CACHE_VERSION or str(z["data_key"]) != data_key:
                logger.info(f"  [{name}] OOF cache ignored (stale: different data or version)")
                return None
            return {
                "oof_proba": z["oof_proba"],
                "fold_aucs": list(z["fold_aucs"]),
                "next_fold": int(z["next_fold"]),
            }
    except Exception:
        logger.warning(f"  [{name}] OOF cache unreadable - starting fresh", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Metrics Utilities
# ---------------------------------------------------------------------------
def compute_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """Compute Kolmogorov-Smirnov (KS) statistic and optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    ks_values = tpr - fpr
    best_idx = np.argmax(ks_values)
    ks_stat = float(ks_values[best_idx])
    best_threshold = float(thresholds[best_idx])
    return ks_stat, best_threshold


def _classification_rates(cm: List[List[int]]) -> Dict[str, float]:
    """Derive accuracy-style metrics from a [[TN, FP], [FN, TP]] matrix.

    With an ~8% positive rate, raw accuracy is dominated by the negative class
    (flagging nobody scores 0.919), so balanced accuracy and MCC are included
    as the honest misclassification measures.
    """
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total = max(tn + fp + fn + tp, 1)
    accuracy = (tp + tn) / total
    pos = tp + fn
    neg = tn + fp
    tpr = tp / pos if pos else 0.0
    tnr = tn / neg if neg else 0.0
    balanced_acc = 0.5 * (tpr + tnr)
    mcc_den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = 0.0 if mcc_den == 0 else ((tp * tn) - (fp * fn)) / mcc_den
    return {
        "accuracy": round(float(accuracy), 6),
        "error_rate": round(float(1.0 - accuracy), 6),
        "balanced_accuracy": round(float(balanced_acc), 6),
        "mcc": round(float(mcc), 6),
    }


def evaluate_predictions(
    y_true: np.ndarray, y_proba: np.ndarray
) -> Dict[str, Any]:
    """Calculate complete suite of credit scoring metrics."""
    auc_roc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))
    ks_stat, opt_thresh = compute_ks_statistic(y_true, y_proba)

    # Metrics at default 0.5 threshold
    y_pred_05 = (y_proba >= 0.5).astype(int)
    f1_05 = float(f1_score(y_true, y_pred_05, zero_division=0))
    prec_05 = float(precision_score(y_true, y_pred_05, zero_division=0))
    rec_05 = float(recall_score(y_true, y_pred_05, zero_division=0))
    cm_05 = confusion_matrix(y_true, y_pred_05).tolist()

    # Metrics at optimal KS threshold
    y_pred_opt = (y_proba >= opt_thresh).astype(int)
    f1_opt = float(f1_score(y_true, y_pred_opt, zero_division=0))
    prec_opt = float(precision_score(y_true, y_pred_opt, zero_division=0))
    rec_opt = float(recall_score(y_true, y_pred_opt, zero_division=0))
    cm_opt = confusion_matrix(y_true, y_pred_opt).tolist()

    return {
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "ks_stat": ks_stat,
        "optimal_threshold": opt_thresh,
        "default_threshold": {
            "threshold": 0.5,
            "f1": f1_05,
            "precision": prec_05,
            "recall": rec_05,
            "confusion_matrix": cm_05,
            **_classification_rates(cm_05),
        },
        "optimal_threshold_metrics": {
            "threshold": opt_thresh,
            "f1": f1_opt,
            "precision": prec_opt,
            "recall": rec_opt,
            "confusion_matrix": cm_opt,
            **_classification_rates(cm_opt),
        },
    }


# ---------------------------------------------------------------------------
# Deep Neural Network (PyTorch)
# ---------------------------------------------------------------------------
class CreditDefaultDNN(nn.Module):
    """Deep Neural Network architecture for tabular default risk."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class PyTorchDNNWrapper:
    """Scikit-learn compatible wrapper for training and evaluating CreditDefaultDNN."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 15,
        device: Optional[str] = None,
        pos_weight: float = 1.0,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.pos_weight = pos_weight

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[CreditDefaultDNN] = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "PyTorchDNNWrapper":
        self.model = CreditDefaultDNN(self.input_dim, self.hidden_dims, self.dropout).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], device=self.device))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=(self.device.type == "cuda"))

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(batch_x)
            scheduler.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        self.model.eval()
        loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=self.batch_size * 2, shuffle=False)
        probas = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                p1 = torch.sigmoid(logits).cpu().numpy()
                probas.append(p1)
        p1_arr = np.concatenate(probas)
        return np.column_stack([1.0 - p1_arr, p1_arr])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Estimator Factories
# ---------------------------------------------------------------------------
def instantiate_estimator(
    name: str,
    best_params: Dict[str, Any],
    spw: float,
    class_weights: Dict[int, float],
    input_dim: int = 246,
    epochs: int = 15,
) -> Any:
    """Instantiate model with tuned hyperparameters and imbalance settings."""
    if name == "logistic_regression":
        cw = best_params.get("class_weight", None)
        if isinstance(cw, str) and cw == "none":
            cw = None
        return LogisticRegression(
            C=best_params.get("C", 1.0),
            penalty=best_params.get("penalty", "l2"),
            solver=best_params.get("solver", "lbfgs"),
            class_weight=cw if cw is not None else class_weights,
            max_iter=2000,
            random_state=RANDOM_STATE,
        )

    if name == "lda":
        return LinearDiscriminantAnalysis(
            solver=best_params.get("solver", "lsqr"),
            shrinkage=best_params.get("shrinkage", "auto"),
            n_components=best_params.get("n_components", None),
        )

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(best_params.get("n_estimators", 150)),
            max_depth=int(best_params.get("max_depth", 14)),
            min_samples_leaf=int(best_params.get("min_samples_leaf", 40)),
            max_features=best_params.get("max_features", "sqrt"),
            class_weight=class_weights,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=int(best_params.get("n_estimators", 150)),
            learning_rate=float(best_params.get("learning_rate", 0.08)),
            max_depth=int(best_params.get("max_depth", 3)),
            subsample=float(best_params.get("subsample", 0.7)),
            random_state=RANDOM_STATE,
        )

    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=int(best_params.get("n_estimators", 500)),
            max_depth=int(best_params.get("max_depth", 5)),
            learning_rate=float(best_params.get("learning_rate", 0.05)),
            subsample=float(best_params.get("subsample", 0.8)),
            colsample_bytree=float(best_params.get("colsample_bytree", 0.8)),
            reg_alpha=float(best_params.get("reg_alpha", 1.0)),
            reg_lambda=float(best_params.get("reg_lambda", 3.0)),
            scale_pos_weight=spw,
            tree_method="hist",
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=int(best_params.get("n_estimators", 500)),
            max_depth=int(best_params.get("max_depth", 7)),
            num_leaves=int(best_params.get("num_leaves", 50)),
            learning_rate=float(best_params.get("learning_rate", 0.05)),
            subsample=float(best_params.get("subsample", 0.8)),
            subsample_freq=1,
            colsample_bytree=float(best_params.get("colsample_bytree", 0.8)),
            reg_alpha=float(best_params.get("reg_alpha", 1.0)),
            reg_lambda=float(best_params.get("reg_lambda", 3.0)),
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

    if name == "dnn":
        return PyTorchDNNWrapper(
            input_dim=input_dim,
            hidden_dims=[256, 128, 64],
            dropout=0.3,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=512,
            epochs=epochs,
            pos_weight=spw,
        )

    raise ValueError(f"Unknown model name: {name}")


# ---------------------------------------------------------------------------
# Training & Cross-Validation Engine
# ---------------------------------------------------------------------------
def train_and_eval_model(
    name: str,
    model_obj: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = CV_FOLDS,
    fold_start: int = 1,
    fold_end: Optional[int] = None,
    resume_state: Optional[Dict[str, Any]] = None,
    defer_final_fit: bool = False,
    models_dir: Optional[Path] = None,
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    """Run stratified CV for OOF evaluation, then fit on full train.

    Chunked/resumable: OOF predictions and completed-fold state are persisted
    under models/oof_cache/ after every fold, so an interrupted or deliberately
    chunked run (--fold-end) continues where it stopped. fold_start/fold_end
    are 1-based inclusive bounds; only folds in that window are processed.
    """
    logger.info(f"--- Training & Evaluating: {name} ({cv_folds}-Fold Stratified CV) ---")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    all_splits = list(skf.split(X_train, y_train))

    # ---- restore or initialize the persisted OOF state
    dk = _data_key(X_train)
    oof_proba = np.zeros(len(y_train), dtype=np.float32)
    fold_aucs: List[float] = []
    next_fold = 1
    cache = resume_state if resume_state is not None else (
        load_oof_cache(models_dir, name, dk) if models_dir is not None else None
    )
    if cache is not None:
        oof_proba = cache["oof_proba"].astype(np.float32)
        fold_aucs = list(cache["fold_aucs"])
        next_fold = int(cache["next_fold"])
        logger.info(f"  [{name}] Resuming OOF cache: folds already done = {next_fold - 1}")

    fold_end_eff = cv_folds if fold_end is None else min(fold_end, cv_folds)
    for fold, (train_idx, val_idx) in enumerate(all_splits, 1):
        if fold < fold_start or fold > fold_end_eff or fold < next_fold:
            continue
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        # Clone / instantiate fold model
        if name == "dnn":
            fold_estimator = PyTorchDNNWrapper(
                input_dim=model_obj.input_dim,
                hidden_dims=model_obj.hidden_dims,
                dropout=model_obj.dropout,
                lr=model_obj.lr,
                weight_decay=model_obj.weight_decay,
                batch_size=model_obj.batch_size,
                epochs=model_obj.epochs,
                pos_weight=model_obj.pos_weight,
            )
        else:
            from sklearn.base import clone

            fold_estimator = clone(model_obj)

        t0 = time.time()
        fold_estimator.fit(X_tr, y_tr)
        val_pred = fold_estimator.predict_proba(X_va)[:, 1]
        oof_proba[val_idx] = val_pred
        fold_auc = roc_auc_score(y_va, val_pred)
        fold_aucs.append(fold_auc)
        next_fold = fold + 1
        if models_dir is not None:
            save_oof_cache(models_dir, name, oof_proba, fold_aucs, next_fold, dk)
        logger.info(f"  [{name}] Fold {fold}/{cv_folds} AUC: {fold_auc:.4f} ({time.time() - t0:.1f}s)")

    metrics = evaluate_predictions(y_train, oof_proba)
    metrics["cv_fold_aucs"] = fold_aucs
    metrics["cv_auc_mean"] = float(np.mean(fold_aucs))
    metrics["cv_auc_std"] = float(np.std(fold_aucs))
    logger.info(
        f"  [{name}] OOF AUC: {metrics['auc_roc']:.4f} (Mean Fold: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}) | KS: {metrics['ks_stat']:.4f}"
    )

    metrics["oof_folds_done"] = next_fold - 1
    if defer_final_fit:
        metrics["fit_time_sec"] = 0.0
        return None, oof_proba, metrics

    # Fit final model on entire training dataset
    logger.info(f"  [{name}] Fitting final model on full dataset ({len(X_train)} samples)...")
    t0 = time.time()
    model_obj.fit(X_train, y_train)
    fit_time = time.time() - t0
    metrics["fit_time_sec"] = round(fit_time, 2)
    logger.info(f"  [{name}] Final model fitted in {fit_time:.2f}s")

    return model_obj, oof_proba, metrics


def build_stacking_ensemble(
    base_models: Dict[str, Any],
    oof_predictions: Dict[str, np.ndarray],
    y_train: np.ndarray,
    X_train_unscaled: np.ndarray,
    X_train_scaled: np.ndarray,
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    """Build and evaluate a Stacking Classifier meta-learner over base models."""
    logger.info("--- Building Stacking Meta-Learner ---")
    model_keys = list(oof_predictions.keys())
    # Stack OOF prediction probabilities as meta-features
    meta_X = np.column_stack([oof_predictions[k] for k in model_keys])

    # Fit logistic regression meta-learner with 5-fold CV
    meta_learner = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    stack_oof = np.zeros(len(y_train), dtype=np.float32)

    fold_aucs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(meta_X, y_train), 1):
        m_tr, y_tr = meta_X[train_idx], y_train[train_idx]
        m_va, y_va = meta_X[val_idx], y_train[val_idx]

        f_meta = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", random_state=RANDOM_STATE)
        f_meta.fit(m_tr, y_tr)
        val_pred = f_meta.predict_proba(m_va)[:, 1]
        stack_oof[val_idx] = val_pred
        f_auc = roc_auc_score(y_va, val_pred)
        fold_aucs.append(f_auc)

    meta_learner.fit(meta_X, y_train)
    metrics = evaluate_predictions(y_train, stack_oof)
    metrics["cv_fold_aucs"] = fold_aucs
    metrics["cv_auc_mean"] = float(np.mean(fold_aucs))
    metrics["cv_auc_std"] = float(np.std(fold_aucs))
    metrics["base_models"] = model_keys
    logger.info(
        f"  [Stacking Ensemble] OOF AUC: {metrics['auc_roc']:.4f} (Mean Fold: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}) | KS: {metrics['ks_stat']:.4f}"
    )

    stack_bundle = {
        "meta_learner": meta_learner,
        "base_model_names": model_keys,
    }
    return stack_bundle, stack_oof, metrics


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate credit scoring models.")
    parser.add_argument("--data-path", type=str, default="data/feature_table.parquet")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="List of models to train: lr lda rf gb xgb lgbm dnn stack all",
    )
    parser.add_argument("--subsample", type=int, default=0, help="Subsample rows for fast runs (0=full)")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for PyTorch DNN")
    parser.add_argument("--fold-start", type=int, default=1, help="1-based first CV fold to train (chunked runs)")
    parser.add_argument("--fold-end", type=int, default=None, help="1-based last CV fold to train (chunked runs)")
    parser.add_argument(
        "--defer-final-fit",
        action="store_true",
        help="Skip the final full-data fit (OOF folds only); final fits then run via --finalize",
    )
    parser.add_argument(
        "--build-stack-from-cache",
        action="store_true",
        help="Assemble the stacking ensemble + benchmark report from cached OOF predictions",
    )
    args = parser.parse_args()
    defer_final_fit = args.defer_final_fit or (args.fold_end is not None and args.fold_end < CV_FOLDS)
    fold_start = args.fold_start
    fold_end = args.fold_end
    resume_state = None  # per-model cache is auto-loaded from models/oof_cache/

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)

    # 1. Load tuning results
    tuning_file = models_dir / "best_hyperparameters.json"
    if not tuning_file.exists():
        logger.error(f"Cannot find {tuning_file}. Please run tune_hyperparameters.py first.")
        sys.exit(1)

    with open(tuning_file, "r") as f:
        tuning_cfg = json.load(f)

    spw = float(tuning_cfg["imbalance"]["scale_pos_weight"])
    cw_raw = tuning_cfg["imbalance"]["class_weight"]
    class_weights = {0: float(cw_raw["0"]), 1: float(cw_raw["1"])}
    tuned_models = tuning_cfg["models"]

    # 2. Load preprocessors
    pre_scaled = joblib.load(models_dir / "preprocessor_scaled.joblib")
    pre_unscaled = joblib.load(models_dir / "preprocessor_unscaled.joblib")

    # 3. Load feature table
    logger.info(f"Loading feature table from {data_path}...")
    df = pd.read_parquet(data_path)
    train_df = df[df["SPLIT"] == "train"].reset_index(drop=True)
    y_full = train_df[TARGET].astype(np.int8).values
    X_full = train_df.drop(columns=[TARGET, "SPLIT"])

    if args.subsample > 0 and args.subsample < len(train_df):
        logger.info(f"Subsampling to {args.subsample} rows (stratified)...")
        # Stratified sampling
        rng = np.random.RandomState(RANDOM_STATE)
        idx_0 = np.where(y_full == 0)[0]
        idx_1 = np.where(y_full == 1)[0]
        sub_0 = rng.choice(idx_0, size=int(args.subsample * (1 - y_full.mean())), replace=False)
        sub_1 = rng.choice(idx_1, size=int(args.subsample * y_full.mean()), replace=False)
        sub_idx = np.sort(np.concatenate([sub_0, sub_1]))
        X_df = X_full.iloc[sub_idx].reset_index(drop=True)
        y = y_full[sub_idx]
    else:
        logger.info(f"Using full training set: {len(X_full):,} rows")
        X_df = X_full
        y = y_full

    # Free the raw table copies early (X_df is what the transforms consume)
    del df, train_df
    import gc

    gc.collect()

    # Models selection
    model_map = {
        "lr": "logistic_regression",
        "lda": "lda",
        "rf": "random_forest",
        "gb": "gradient_boosting",
        "xgb": "xgboost",
        "lgbm": "lightgbm",
        "dnn": "dnn",
    }

    if "all" in args.models:
        selected_models = ["lr", "lda", "rf", "gb", "xgb", "lgbm", "dnn"]
    else:
        selected_models = []
        for m in args.models:
            if m in model_map:
                selected_models.append(m)
            elif m in model_map.values():
                selected_models.append(list(model_map.keys())[list(model_map.values()).index(m)])

    # Transform only what the selected models need (halves peak RAM for
    # tree-only or linear-only launches).
    scaled_names = ("logistic_regression", "lda", "dnn")
    needs_scaled = any(model_map[m] in scaled_names for m in selected_models)
    needs_unscaled = any(model_map[m] not in scaled_names for m in selected_models)

    logger.info("Pre-transforming feature matrices...")
    X_scaled = transform_f32(pre_scaled, X_df) if needs_scaled else None
    X_unscaled = transform_f32(pre_unscaled, X_df) if needs_unscaled else None

    num_features = (X_scaled if X_scaled is not None else X_unscaled).shape[1]
    logger.info(f"Input feature dimensionality: {num_features}")

    oof_dict: Dict[str, np.ndarray] = {}
    trained_models: Dict[str, Any] = {}
    all_metrics: Dict[str, Any] = {}
    build_stack = args.build_stack_from_cache or "all" in args.models or "stack" in args.models

    if args.build_stack_from_cache:
        # Assemble stacking + report from persisted OOFs: needs no base model in
        # memory and no training this launch (all folds already cached).
        _probe = pre_unscaled.transform(X_df.iloc[:2])
        if hasattr(_probe, "toarray"):
            _probe = _probe.toarray()
        dk_full = f"{len(y_full)}x{np.asarray(_probe).shape[1]}"
        stack_model_keys = [
            "logistic_regression", "lda", "random_forest",
            "xgboost", "lightgbm", "dnn",
        ]
        for k in stack_model_keys:
            c = load_oof_cache(models_dir, k, dk_full)
            if c is None or int(c["next_fold"]) <= CV_FOLDS:
                done = 0 if c is None else int(c["next_fold"]) - 1
                logger.error(
                    "OOF cache for '%s' missing/incomplete (%d/5 folds) - finish base models first.",
                    k, done,
                )
                sys.exit(1)
            oof_dict[k] = c["oof_proba"].astype(np.float32)
            logger.info("  [stack] using cached OOF for %s", k)
        # merge per-model metrics accumulated by the chunked training launches
        chunk_path = models_dir / "chunk_metrics.json"
        if chunk_path.exists():
            try:
                for k, v in json.loads(chunk_path.read_text(encoding="utf-8")).items():
                    all_metrics.setdefault(k, v)
            except Exception:
                logger.warning("Could not merge chunk_metrics.json", exc_info=True)
        selected_models = []

    for alias in selected_models:
        m_name = model_map[alias]
        b_params = tuned_models.get(m_name, {}).get("best_params", {})
        estimator = instantiate_estimator(
            name=m_name,
            best_params=b_params,
            spw=spw,
            class_weights=class_weights,
            input_dim=num_features,
            epochs=args.epochs,
        )

        # Scaled vs unscaled assignment
        if m_name in ("logistic_regression", "lda", "dnn"):
            X_curr = X_scaled
        else:
            X_curr = X_unscaled

        # Fit the final production model on-demand (per-process) so a chunked
        # run only pays this cost in the launch that actually needs it. Boosters
        # take minutes at 307k rows, so they get dedicated invocations.
        # (defer_final_fit returns fitted_est=None; uniform 3-tuple unpack.)
        fitted_est, oof_p, metrics = train_and_eval_model(
            name=m_name,
            model_obj=estimator,
            X_train=X_curr,
            y_train=y,
            fold_start=fold_start,
            fold_end=fold_end,
            resume_state=resume_state,
            defer_final_fit=defer_final_fit,
            models_dir=models_dir,
        )
        oof_dict[m_name] = oof_p
        trained_models[m_name] = fitted_est
        all_metrics[m_name] = metrics

        # Persist model
        if m_name == "dnn":
            if fitted_est is not None:
                torch.save(fitted_est.model.state_dict(), models_dir / "dnn_model.pt")
                logger.info(f"Saved PyTorch DNN weights to {models_dir / 'dnn_model.pt'}")
            else:
                logger.info(f"[{m_name}] final fit deferred - weights not saved this launch")
        elif fitted_est is not None:
            joblib.dump(fitted_est, models_dir / f"{m_name}.joblib")
            logger.info(f"Saved {m_name} to {models_dir / f'{m_name}.joblib'}")

    # Build Stacking meta-learner if requested and base OOFs are complete
    complete_oofs = {
        k: oof_dict[k]
        for k in oof_dict
        if all_metrics.get(k, {}).get("oof_folds_done", CV_FOLDS) >= CV_FOLDS
    }
    if build_stack and "stacking_ensemble" not in all_metrics and len(complete_oofs) >= 2:
        stack_bundle, stack_oof, stack_metrics = build_stacking_ensemble(
            base_models=trained_models,
            oof_predictions=complete_oofs,
            y_train=y,
            X_train_unscaled=X_unscaled,
            X_train_scaled=X_scaled,
        )
        oof_dict["stacking_ensemble"] = stack_oof
        all_metrics["stacking_ensemble"] = stack_metrics
        joblib.dump(stack_bundle, models_dir / "stacking_ensemble.joblib")
        logger.info(f"Saved Stacking Ensemble to {models_dir / 'stacking_ensemble.joblib'}")
    elif build_stack and "stacking_ensemble" not in all_metrics:
        logger.warning(
            "Stacking requested but only %d model(s) have complete OOF predictions - skipping.",
            len(complete_oofs),
        )

    # Accumulate per-launch metrics for chunked runs. evaluation_results.json is
    # only rewritten once every model (and the stack) is complete, so partial
    # chunks never clobber a full benchmark.
    if all_metrics:
        chunk_path = models_dir / "chunk_metrics.json"
        merged: Dict[str, Any] = {}
        if chunk_path.exists():
            try:
                merged = json.loads(chunk_path.read_text(encoding="utf-8"))
            except Exception:
                merged = {}
        merged.update(all_metrics)
        chunk_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        logger.info("Updated %s (%d model entries total)", chunk_path, len(merged))

        all_complete = (
            "stacking_ensemble" in merged
            and all(
                m.get("oof_folds_done", CV_FOLDS) >= CV_FOLDS
                for k, m in merged.items()
                if k != "stacking_ensemble"
            )
        )
        if all_complete:
            summary_path = models_dir / "evaluation_results.json"
            with open(summary_path, "w") as f:
                json.dump(merged, f, indent=2)
            logger.info(f"Saved evaluation metrics to {summary_path}")

    # 5. Generate Markdown Comparison Table
    md_lines = [
        "# Model Comparison & Evaluation Results",
        "",
        f"> **Evaluated at**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> **Cross-Validation**: 5-Fold Stratified CV (Out-Of-Fold Evaluation)",
        f"> **Training Samples**: {len(y):,}",
        "",
        "## Overall Benchmark",
        "",
        "| Model | CV AUC Mean ± Std | OOF AUC-ROC | PR-AUC | KS-Statistic | F1 (at 0.5) | F1 (Opt Thresh) | Optimal Thresh |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for model_name, m_data in all_metrics.items():
        mean_cv = f"{m_data.get('cv_auc_mean', 0.0):.4f} ± {m_data.get('cv_auc_std', 0.0):.4f}"
        auc = f"{m_data.get('auc_roc', 0.0):.4f}"
        pr = f"{m_data.get('pr_auc', 0.0):.4f}"
        ks = f"{m_data.get('ks_stat', 0.0):.4f}"
        f1_05 = f"{m_data.get('default_threshold', {}).get('f1', 0.0):.4f}"
        f1_opt = f"{m_data.get('optimal_threshold_metrics', {}).get('f1', 0.0):.4f}"
        opt_t = f"{m_data.get('optimal_threshold', 0.5):.4f}"
        md_lines.append(
            f"| `{model_name}` | {mean_cv} | **{auc}** | {pr} | **{ks}** | {f1_05} | **{f1_opt}** | {opt_t} |"
        )

    md_lines.append("")
    md_lines.append("## Key Insights & Discussion")
    md_lines.append("")
    md_lines.append("- **Class Imbalance Impact**: Notice the stark difference between default 0.5 threshold F1 and optimal KS threshold F1. Due to the 11.4:1 class imbalance, threshold tuning is essential for non-weighted estimators.")
    md_lines.append("- **Ensemble Superiority**: Gradient Boosted trees (LightGBM / XGBoost) and the Stacking Ensemble achieve the highest AUC and KS discrimination.")
    md_lines.append("- **Artifacts Saved**: All models are persisted in `models/` for real-time inference via the API server.")
    md_lines.append("")

    report_md_path = models_dir / "evaluation_results.md"
    if all_metrics and "stacking_ensemble" in all_metrics:
        report_md_path.write_text("\n".join(md_lines), encoding="utf-8")
        logger.info(f"Saved benchmark markdown report to {report_md_path}")
        print("\n" + "\n".join(md_lines))
    else:
        logger.info("Benchmark report left unchanged (stacking ensemble not rebuilt this launch).")


if __name__ == "__main__":
    main()
