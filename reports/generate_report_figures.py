"""Generate the report's Chapter-6 figures from the real trained artifacts.

Sources (all real, no placeholders):
  models/evaluation_results.json  -- OOF/CV benchmark for the models
  models/oof_cache/*.npz          -- per-model out-of-fold probabilities ('oof_proba')
  data/feature_table.parquet      -- train/test rows ('SPLIT' column) for the SHAP sample
  models/lightgbm.joblib          -- champion model for exact TreeSHAP (pred_contrib=True)

Usage:
  python reports/generate_report_figures.py

Outputs (PNG, 300 DPI) under reports/figures/:
  fig6_1_model_benchmark.png   4-panel horizontal bars: AUC / KS / F1@0.5 / F1@opt
  fig6_2_roc_curves.png        OOF ROC curves for all models
  fig6_3_ks_curves.png         KS separation curves (CDFs of score by class)
  fig6_4_shap_beeswarm.png     TreeSHAP beeswarm + mean |SHAP| top-20 for LightGBM
"""

from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")
OUT_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

# (artifact key, display label, family color)
MODELS = [
    ("logistic_regression", "Logistic Regression", "#8FA9C4"),
    ("lda", "LDA", "#B3C6E7"),
    ("gradient_boosting", "Gradient Boosting", "#8CD9B3"),
    ("random_forest", "Random Forest", "#5FBD8F"),
    ("xgboost", "XGBoost", "#E6A23C"),
    ("lightgbm", "LightGBM", "#D9534F"),
    ("dnn", "DNN (MLP)", "#B085C9"),
    ("stacking_ensemble", "Stacking Ensemble", "#0D3B36"),
]
ORDER = [k for k, _, _ in MODELS]
LABEL = {k: lbl for k, lbl, _ in MODELS}
COLOR = {k: c for k, _, c in MODELS}

TEAL = "#0D3B36"


def load_results() -> dict:
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_oof(key: str):
    """Return (y_true, y_proba) OOF arrays for a model from oof_cache, or None.

    The training pipeline saves only 'oof_proba' + 'fold_aucs' (labels are not
    stored per model), so the shared TARGET column is loaded separately by the
    caller and passed in.
    """
    path = os.path.join(MODELS_DIR, "oof_cache", f"{key}.npz")
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if "oof_proba" not in z.files:
            return None
        return np.asarray(z["oof_proba"], dtype=np.float64)
    except Exception as exc:
        print(f"  ! could not read OOF cache for {key}: {exc}")
        return None


def load_targets():
    """Load the training TARGET column (aligned with OOF cache order)."""
    import pandas as pd

    df = pd.read_parquet(
        os.path.join(ROOT, "data", "feature_table.parquet"),
        columns=["TARGET", "SPLIT"],
    )
    train = df[df["SPLIT"] == "train"]
    return train["TARGET"].astype(np.int8).to_numpy()


# ---------------------------------------------------------------- Figure 6.1
def fig_benchmark(results: dict) -> None:
    rows = []
    for key in ORDER:
        m = results.get(key)
        if not m:
            continue
        rows.append(
            {
                "key": key,
                "Model": LABEL[key],
                "AUC-ROC": m["auc_roc"],
                "KS": m["ks_stat"],
                "F1@0.5": m["default_threshold"]["f1"],
                "F1@opt": m["optimal_threshold_metrics"]["f1"],
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    fig.suptitle(
        "Figure 6.1  Comparative performance benchmark (5-fold OOF, n = 307,511)",
        fontsize=12.5,
        y=0.99,
    )

    def hbar(ax, col, title, xlabel, xlim, fmt="{:.4f}"):
        d = sorted(rows, key=lambda r: r[col])
        y = np.arange(len(d))
        colors = [COLOR[r["key"]] for r in d]
        ax.barh(y, [r[col] for r in d], height=0.62, color=colors,
                edgecolor="#334155", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([r["Model"] for r in d], fontsize=8.5)
        ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        span = xlim[1] - xlim[0]
        for rect, r in zip(ax.patches, d):
            ax.text(rect.get_width() + span * 0.012,
                    rect.get_y() + rect.get_height() / 2,
                    fmt.format(r[col]), va="center", fontsize=7.6)

    hbar(axes[0][0], "AUC-ROC", "(a) Discrimination", "AUC-ROC", (0.72, 0.80))
    hbar(axes[0][1], "KS", "(b) Class separation", "KS statistic", (0.36, 0.45))
    hbar(axes[1][0], "F1@0.5", "(c) F1 at default threshold p = 0.50", "F1-score", (0.0, 0.34), fmt="{:.3f}")
    hbar(axes[1][1], "F1@opt", "(d) F1 at optimal (KS-max) threshold", "F1-score", (0.25, 0.30))

    fig.text(0.5, 0.005,
             "Stacking Ensemble and LightGBM lead on AUC/KS. Panel (c) illustrates the "
             "threshold-sensitivity trap: F1@0.5 collapses for models whose optimal "
             "threshold sits far below 0.5.",
             ha="center", fontsize=8, style="italic")

    fig.savefig(os.path.join(OUT_DIR, "fig6_1_model_benchmark.png"))
    plt.close(fig)
    print("wrote fig6_1_model_benchmark.png")


# ------------------------------------------------- Figures 6.2 (ROC) & 6.3 (KS)
def fig_roc(y_true: np.ndarray, probas: dict, results: dict) -> None:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    diag = np.linspace(0, 1, 100)
    ax.plot(diag, diag, color="#94a3b8", ls=":", lw=1.1, label="Chance (AUC 0.500)")

    for key in ORDER:
        if key not in probas:
            continue
        m = results.get(key, {})
        fpr, tpr, _ = roc_curve(y_true, probas[key])
        emph = key in ("stacking_ensemble", "lightgbm")
        ax.plot(fpr, tpr, color=COLOR[key], lw=1.9 if emph else 1.2,
                label=f"{LABEL[key]}  (AUC {m.get('auc_roc', 0):.4f})", zorder=3 if emph else 2)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Figure 6.2  OOF ROC curves (5-fold stratified CV, n = 307,511)", fontsize=11)
    ax.legend(fontsize=7.8, loc="lower right", framealpha=0.95)
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.005)

    fig.savefig(os.path.join(OUT_DIR, "fig6_2_roc_curves.png"))
    plt.close(fig)
    print("wrote fig6_2_roc_curves.png")


def fig_ks(y_true: np.ndarray, probas: dict, results: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    bins = np.linspace(0.0, 1.0, 401)

    ks_lines = []
    for key in ORDER:
        if key not in probas:
            continue
        p = probas[key]
        p0, p1 = p[y_true == 0], p[y_true == 1]
        cdf0 = np.searchsorted(np.sort(p0), bins, side="right") / max(len(p0), 1)
        cdf1 = np.searchsorted(np.sort(p1), bins, side="right") / max(len(p1), 1)
        gap = np.abs(cdf1 - cdf0)
        ks_at = bins[int(np.argmax(gap))]
        m = results.get(key, {})
        ks_lines.append((gap.max(), LABEL[key], ks_at, m.get("ks_stat", 0)))
        emph = key in ("stacking_ensemble", "lightgbm")
        ax.plot(bins, cdf0, color=COLOR[key], lw=1.9 if emph else 1.0,
                label=f"{LABEL[key]} — non-default CDF", zorder=3 if emph else 2)
        ax.plot(bins, cdf1, color=COLOR[key], lw=1.9 if emph else 1.0, ls="--",
                zorder=3 if emph else 2)

    # annotate the KS gap for the two champions
    for ks_val, lbl, ks_at, _ in sorted(ks_lines, reverse=True)[:2]:
        ax.annotate(f"KS = {ks_val:.3f}", xy=(ks_at, 0.5),
                    xytext=(ks_at + 0.06, 0.38), fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="#334155", lw=0.8))

    ax.set_xlabel("Predicted default probability (score)")
    ax.set_ylabel("Cumulative fraction of population")
    ax.set_title("Figure 6.3  KS separation: score CDFs, default (dashed) vs non-default (solid)",
                 fontsize=10.5)
    ax.set_xlim(0, 0.85)
    ax.legend(fontsize=6.6, loc="upper right", ncol=2, framealpha=0.95)
    ax.text(0.985, 0.02, "solid = class 0 (repaid)   dashed = class 1 (default)",
            transform=ax.transAxes, ha="right", fontsize=7.5, style="italic")

    fig.savefig(os.path.join(OUT_DIR, "fig6_3_ks_curves.png"))
    plt.close(fig)
    print("wrote fig6_3_ks_curves.png")


# ---------------------------------------------------------------- Figure 6.4
def fig_shap(results: dict) -> None:
    """Exact TreeSHAP via LightGBM's pred_contrib=True (mirrors the API path;
    the external `shap` package is not installed in this environment)."""
    import joblib
    import pandas as pd

    model_path = os.path.join(MODELS_DIR, "lightgbm.joblib")
    if not os.path.exists(model_path):
        print("fig6_4 skipped: models/lightgbm.joblib not found")
        return
    model = joblib.load(model_path)

    # Build the exact transformed matrix the API serves: 207 raw cols -> 259
    df = pd.read_parquet(os.path.join(ROOT, "data", "feature_table.parquet"))
    train = df[df["SPLIT"] == "train"].reset_index(drop=True)
    y = train["TARGET"].astype(np.int8).to_numpy()
    raw = train.drop(columns=[c for c in ("TARGET", "SPLIT") if c in train.columns])

    pre_path = os.path.join(MODELS_DIR, "preprocessor_scaled.joblib")
    if not os.path.exists(pre_path):
        print("fig6_4 skipped: preprocessor_scaled.joblib not found")
        return
    pre = joblib.load(pre_path)

    # stratified sample of 2,000 training rows
    rng = np.random.default_rng(42)
    idx0 = rng.choice(np.where(y == 0)[0], size=1560, replace=False)
    idx1 = rng.choice(np.where(y == 1)[0], size=440, replace=False)
    idx = np.sort(np.concatenate([idx0, idx1]))
    Xs = pre.transform(raw.iloc[idx])

    contribs = model.predict(Xs, pred_contrib=True)  # (n, n_features + 1)
    sv = contribs[:, :-1]
    base = float(contribs[0, -1])
    print(f"  TreeSHAP base value = {base:.4f} over {Xs.shape[1]} features, n = {len(Xs)}")

    mean_abs = np.abs(sv).mean(axis=0)
    top = np.argsort(mean_abs)[::-1][:20]

    names = None
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        pass
    if names is None or len(names) != Xs.shape[1]:
        names = [f"f{i}" for i in range(Xs.shape[1])]
    top_names = [names[i].replace("num__", "").replace("cat__", "") for i in top]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.8), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("Figure 6.4  TreeSHAP explainability — champion LightGBM "
                 "(2,000-row stratified sample, exact pred_contrib attributions)",
                 fontsize=11.5, y=1.0)

    # (a) beeswarm
    ax = axes[0]
    rngj = np.random.default_rng(7)
    Xs_dense = np.asarray(Xs.todense()) if hasattr(Xs, "todense") else np.asarray(Xs)
    for rank, fi in enumerate(top):
        v = sv[:, fi]
        std = Xs_dense[:, fi].std()
        color_vals = (Xs_dense[:, fi] - Xs_dense[:, fi].mean()) / (std or 1.0)
        order = np.argsort(np.abs(v))
        jitter = (rngj.random(len(v)) - 0.5) * 0.62
        ax.scatter(v[order], rank + jitter[order], c=color_vals[order],
                   cmap="coolwarm", vmin=-2, vmax=2, s=4, alpha=0.5,
                   edgecolors="none", rasterized=True)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top_names, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color="#334155", lw=0.8)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title("(a) Beeswarm: red = high feature value", fontsize=9.5)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    # (b) mean |SHAP| bar chart
    ax2 = axes[1]
    ax2.barh(range(len(top)), mean_abs[top], color=TEAL, edgecolor="#334155", lw=0.5)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels(top_names, fontsize=7.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("mean |SHAP value|")
    ax2.set_title("(b) Global importance (top 20)", fontsize=9.5)
    ax2.grid(axis="x", alpha=0.25, linestyle="--")

    fig.savefig(os.path.join(OUT_DIR, "fig6_4_shap_beeswarm.png"))
    plt.close(fig)
    print("wrote fig6_4_shap_beeswarm.png")


def main() -> None:
    results = load_results()

    y_true = load_targets()
    probas = {}
    for key in ORDER:
        p = load_oof(key)
        if p is not None and len(p) == len(y_true):
            probas[key] = p
        elif p is not None:
            print(f"  ! OOF length mismatch for {key}: {len(p)} vs {len(y_true)}")

    print(f"Loaded OOF probabilities for {len(probas)}/{len(ORDER)} models "
          f"(gradient_boosting has no fold cache; bars only)")

    fig_benchmark(results)
    fig_roc(y_true, probas, results)
    fig_ks(y_true, probas, results)
    fig_shap(results)
    print("All figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
