"""Hyperparameter tuning for the loan default project (6 models).

RandomizedSearchCV with 5-fold StratifiedKFold, scoring='roc_auc',
n_iter capped per model. Produces a best-params config for the training
step - final models are NOT retrained here.

Models tuned here: LogisticRegression, LinearDiscriminantAnalysis,
RandomForest, GradientBoosting, XGBoost, LightGBM.
(Stacking meta-learner and DNN are handled separately.)

Runtime strategy (300k+ rows):
- Tuning runs on a stratified subsample (default 120k rows, preserves the
  ~8% default rate) with full-train training later in the training step.
- n_jobs=1 inside estimators; RandomizedSearchCV parallelizes across
  candidates (n_jobs=8) so 16 cores stay busy without oversubscription.
- Results checkpoint after every model, so an interrupted run resumes.

Usage:
    python server/tune_hyperparameters.py [--subsample 120000] [--n-iter 25]
        [--models lgbm xgb ...] [--full]      # --full: tune on all 307k rows
"""

from __future__ import annotations

import os
# Thread-pinning MUST happen before numpy/sklearn import: with 8 parallel
# candidates x 16-core BLAS, oversubscription made fits ~10x slower.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "roc_auc"
N_JOBS_SEARCH = 8  # candidates in parallel; estimators run n_jobs=1
SUBSAMPLE_DEFAULT = 50_000
N_ITER_DEFAULT = 25

# ---------------------------------------------------------------------------
# model factories + search spaces
# ---------------------------------------------------------------------------
def build_model_specs(spw: float, class_weights: Dict[int, float]) -> Dict[str, Dict[str, Any]]:
    """Model estimator factories and RandomizedSearchCV spaces.

    spw: scale_pos_weight for XGBoost/LightGBM (imbalance handling).
    class_weights: class_weight dict for LR/RF (imbalance handling).
    LDA/GB have no class-weight hooks in sklearn; their imbalance handling
    is deferred to threshold tuning at training time (noted in the config).
    """
    specs: Dict[str, Dict[str, Any]] = {
        "logistic_regression": {
            "estimator": LogisticRegression(
                class_weight=class_weights, max_iter=2000, random_state=RANDOM_STATE
            ),
            "params": {
                # lbfgs+l2 only: measured 17s/fit at 120k rows vs 57s+ for
                # liblinear (and far worse at high C) - l1 not worth the cost
                "C": uniform(loc=0.01, scale=10.0),          # 0.01 .. 10.01
                "penalty": ["l2"],
                "solver": ["lbfgs"],
                "class_weight": [class_weights, None],
            },
            "n_iter": 8,  # only 2 real dimensions (C x class_weight)
        },
        "lda": {
            "estimator": LinearDiscriminantAnalysis(),
            "params": {
                "solver": ["lsqr", "eigen"],
                "shrinkage": ["auto", None, 0.0, 0.25, 0.5, 0.75],
                "n_components": [None, 1],
            },
            "n_iter": 12,  # tiny space, no need for 25
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                class_weight=class_weights, random_state=RANDOM_STATE, n_jobs=1
            ),
            "params": {
                # measured at 120k rows: unbounded depth costs >270s/fit.
                # sqrt/0.2 max_features and <=200 trees keep ~5-fold searches
                # ~12 min; the final training step can still grow deeper trees.
                "n_estimators": randint(80, 160),
                "max_depth": [8, 12, 16],
                "min_samples_leaf": randint(20, 80),
                "max_features": ["sqrt", 0.2],
            },
            "n_iter": 5,
            # RF trees parallelize well: 2 concurrent searches x 4-tree-threads
            # keeps 8 cores busy with far less contention than 8 serial fits
            "search_n_jobs": 2,
            "estimator_n_jobs": 4,
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                # sklearn GB has no histogram speedup (~0.2s/tree at 120k rows) -
                # tight space keeps 5-fold x n_iter within ~25 min
                "n_estimators": randint(60, 160),
                "learning_rate": uniform(loc=0.01, scale=0.19),  # 0.01 .. 0.20
                "max_depth": randint(2, 5),
                "subsample": uniform(loc=0.6, scale=0.35),       # 0.6 .. 0.95
            },
            "n_iter": 6,
            "search_n_jobs": 4,
            "note": "no native class weights; threshold tuning at training time",
        },
        "xgboost": {
            "estimator": None,  # built lazily (import inside function)
            "params": {
                "n_estimators": randint(300, 700),
                "max_depth": randint(3, 10),
                "learning_rate": uniform(loc=0.01, scale=0.19),
                "subsample": uniform(loc=0.6, scale=0.35),
                "colsample_bytree": uniform(loc=0.5, scale=0.45),
                "reg_alpha": uniform(loc=0, scale=5.0),
                "reg_lambda": uniform(loc=0.5, scale=9.5),
            },
            "n_iter": 8,
        },
        "lightgbm": {
            "estimator": None,
            "params": {
                "n_estimators": randint(300, 700),
                "max_depth": randint(4, 12),
                "num_leaves": randint(20, 120),
                "learning_rate": uniform(loc=0.01, scale=0.19),
                "subsample": uniform(loc=0.6, scale=0.35),
                "subsample_freq": [1],
                "colsample_bytree": uniform(loc=0.5, scale=0.45),
                "reg_alpha": uniform(loc=0, scale=5.0),
                "reg_lambda": uniform(loc=0.5, scale=9.5),
            },
            "n_iter": 8,
        },
    }
    return specs


def _make_estimator(name: str, spw: float):
    """Lazily construct boosters (imported here so tests skip if absent)."""
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            eval_metric="auc",
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=-1,
        )
    raise KeyError(name)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_tuning_data(
    parquet_path: Path, subsample: int, full: bool
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the feature table and optionally stratified-subsample train rows."""
    df = pd.read_parquet(parquet_path)
    train = df[df["SPLIT"] == "train"].reset_index(drop=True)
    y = train["TARGET"].astype(np.int8)
    X = train.drop(columns=["TARGET", "SPLIT"])

    if full or subsample >= len(X):
        logger.info("Using FULL train set: %d rows", len(X))
        return X, y

    # stratified subsample preserving the ~8% default rate
    frac = subsample / len(X)
    part = []
    for label in (0, 1):
        sub = train[train["TARGET"] == label]
        part.append(sub.sample(n=max(1, int(round(len(sub) * frac))), random_state=RANDOM_STATE))
    sub_df = pd.concat(part).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    y_sub = sub_df["TARGET"].astype(np.int8)
    X_sub = sub_df.drop(columns=["TARGET", "SPLIT"])
    logger.info(
        "Stratified subsample: %d rows (default rate %.2f%%)",
        len(X_sub),
        100 * y_sub.mean(),
    )
    return X_sub, y_sub


def prepare_matrices(
    X: pd.DataFrame, preprocessor
) -> Tuple[np.ndarray, List[str]]:
    """Transform features with the fitted preprocessing pipeline."""
    Xt = preprocessor.transform(X)
    names = list(preprocessor.get_feature_names_out())
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    return np.asarray(Xt, dtype=np.float32), names


# ---------------------------------------------------------------------------
# search runner
# ---------------------------------------------------------------------------
def run_search(
    name: str,
    spec: Dict[str, Any],
    X: np.ndarray,
    y: pd.Series,
    spw: float,
    n_iter_override: int | None = None,
) -> Dict[str, Any]:
    """Run one RandomizedSearchCV; returns result dict for the config."""
    # explicit None check: `or` would call __len__ on ensemble estimators,
    # which raises AttributeError on unfitted forests (estimators_ missing)
    estimator = spec["estimator"] if spec["estimator"] is not None else _make_estimator(name, spw)
    params = dict(spec["params"])
    n_iter = n_iter_override or spec["n_iter"]

    # per-model parallelism profile (see build_model_specs notes)
    est_n_jobs = spec.get("estimator_n_jobs")
    if est_n_jobs is not None and hasattr(estimator, "n_jobs"):
        estimator.set_params(n_jobs=est_n_jobs)
    search_n_jobs = spec.get("search_n_jobs", N_JOBS_SEARCH)

    # lightgbm reads spw via constructor; xgboost ditto. class_weight models
    # already carry their weights. So param grids never touch spw/class_weight
    # except LR where a [weighted, None] comparison is part of the space.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_iter=n_iter,
        scoring=SCORING,
        cv=cv,
        n_jobs=search_n_jobs,
        verbose=2,
        random_state=RANDOM_STATE,
        refit=False,  # training step does final fits; search only evaluates
        return_train_score=False,
        error_score=np.nan,
    )
    t0 = time.time()
    search.fit(X, y)
    elapsed = time.time() - t0

    # top-5 candidates sorted by mean CV AUC
    pairs = list(zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]))
    pairs.sort(key=lambda t: t[1], reverse=True)

    best = {
        "model": name,
        "best_cv_auc": float(search.best_score_),
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        "n_iter": n_iter,
        "n_candidates": len(search.cv_results_["params"]),
        "elapsed_sec": round(elapsed, 1),
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "scoring": SCORING,
        "top5": [
            {
                "params": {k: _jsonable(v) for k, v in p.items()},
                "mean_auc": float(m),
            }
            for p, m in pairs[:5]
        ],
    }
    return best


def _jsonable(v: Any) -> Any:
    """Make numpy/scipy values JSON-serializable."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def save_results(
    results: Dict[str, Any], out_json: Path, out_md: Path
) -> None:
    out_json.write_text(json.dumps(results, indent=2, default=_jsonable), encoding="utf-8")

    lines = [
        "# Hyperparameter Tuning Results",
        "",
        f"- Search: RandomizedSearchCV, {CV_FOLDS}-fold stratified CV, scoring=roc_auc",
        f"- random_state={RANDOM_STATE} everywhere; final models trained separately",
        "",
        "| Model | Best CV AUC | n_iter | Elapsed (s) | Best Params |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for name, r in sorted(results["models"].items(), key=lambda kv: kv[1]["best_cv_auc"], reverse=True):
        params_str = json.dumps(r["best_params"])
        if len(params_str) > 90:
            params_str = params_str[:87] + "..."
        lines.append(
            f"| {name} | {r['best_cv_auc']:.4f} | {r['n_iter']} | {r['elapsed_sec']} | `{params_str}` |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Tuning data: see `data_size` in best_params JSON.",
        "- LDA / GradientBoosting: no class-weight hook; handle imbalance via",
        "  threshold tuning at training time.",
        "- Stacking meta-learner and DNN tuning handled separately.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %s and %s", out_json, out_md)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Tune 6 models via RandomizedSearchCV.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--outdir", type=str, default="models")
    parser.add_argument("--subsample", type=int, default=SUBSAMPLE_DEFAULT)
    parser.add_argument("--n-iter", type=int, default=None, help="Override n_iter for all models")
    parser.add_argument("--full", action="store_true", help="Tune on the full 307k train rows")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of models to tune (default: all 6)",
    )
    args = parser.parse_args()

    # make sklearn verbose output appear immediately when stdout is a file
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "best_hyperparameters.json"
    md_path = outdir / "tuning_results.md"

    # ---- load preprocessing artifacts + data
    # NOTE: tuning uses the SCALED preprocessor. liblinear/saga converge orders
    # of magnitude faster on standardized features, and scaling does not change
    # tree-model behavior at all (splits are scale-invariant).
    import joblib

    pre = joblib.load(outdir / "preprocessor_scaled.joblib")
    df_path = Path(args.data_dir) / "feature_table.parquet"
    X_raw, y = load_tuning_data(df_path, args.subsample, args.full)
    X, feature_names = prepare_matrices(X_raw, pre)

    # ---- imbalance params from TRAIN y
    from preprocessing import get_class_weights, get_scale_pos_weight

    spw = get_scale_pos_weight(y)
    cw = get_class_weights(y)

    # ---- checkpoint: resume support
    results: Dict[str, Any] = {
        "meta": {
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
            "scoring": SCORING,
            "data_size": int(len(y)),
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
        "imbalance": {"scale_pos_weight": spw, "class_weight": cw},
        "models": {},
    }
    if json_path.exists():
        old = json.loads(json_path.read_text(encoding="utf-8"))
        if old.get("meta", {}).get("data_size") == len(y):
            results["models"] = old.get("models", {})
            logger.info("Resuming: %d model(s) already tuned", len(results["models"]))
        elif args.models:
            # Subset re-tune at a different data size: carry over the models that
            # are NOT being re-tuned (keeping their original data_size marker) so
            # a partial re-tune MERGES into the config instead of silently
            # dropping them. Re-tuned models are written fresh at the new size.
            carried = {
                k: v for k, v in old.get("models", {}).items() if k not in set(args.models)
            }
            results["models"] = carried
            logger.info(
                "Merge mode: re-tuning %s at data_size=%d; carrying over %d model(s) tuned at their original sizes",
                ", ".join(args.models),
                len(y),
                len(carried),
            )
        else:
            logger.warning(
                "Existing config was tuned at data_size=%s but this run uses %d rows; "
                "results will REPLACE it (back up the file first if needed).",
                old.get("meta", {}).get("data_size"),
                len(y),
            )

    specs = build_model_specs(spw, cw)
    wanted = args.models or list(specs.keys())

    for name in wanted:
        if name not in specs:
            logger.error("Unknown model '%s' - skipping", name)
            continue
        if name in results["models"]:
            logger.info("[%s] already tuned - skipping", name)
            continue
        logger.info("=" * 70)
        logger.info("Tuning %s ...", name)
        try:
            res = run_search(name, specs[name], X, y, spw, n_iter_override=args.n_iter)
        except Exception:
            logger.exception("Search failed for %s", name)
            continue
        res["data_size"] = int(len(y))
        results["models"][name] = res
        save_results(results, json_path, md_path)  # checkpoint after each model
        logger.info(
            "[%s] best CV AUC %.4f | %s",
            name,
            res["best_cv_auc"],
            res["best_params"],
        )

    logger.info("All requested models tuned. Config at %s", json_path.resolve())


if __name__ == "__main__":
    main()
