"""Generate publication-quality confusion matrix figures from real OOF data.

Outputs (300 DPI PNG) to reports/figures/:
  fig6_5_confusion_matrices_all.png   -- 2x4 grid: all 7 models at optimal threshold
  fig6_6_confusion_matrix_champion.png -- Single large: Stacking Ensemble
  fig6_7_confusion_matrix_compare.png  -- Side-by-side: Champion vs LR baseline

Usage:
    python reports/generate_confusion_matrices.py
"""

from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(ROOT, "models", "evaluation_results.json")
OUT_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Display order and labels
MODELS = [
    ("stacking_ensemble", "Stacking Ensemble",   "#0D3B36"),
    ("lightgbm",          "LightGBM",            "#D9534F"),
    ("xgboost",           "XGBoost",             "#E6A23C"),
    ("dnn",               "DNN",                 "#B085C9"),
    ("logistic_regression","Logistic Regression", "#8FA9C4"),
    ("random_forest",     "Random Forest",       "#5FBD8F"),
    ("lda",               "LDA",                 "#B3C6E7"),
    ("gradient_boosting", "Gradient Boosting",   "#8CD9B3"),
]

CLASS_LABELS = ["Repaid\n(0)", "Default\n(1)"]
POPULATION = 307_511
DEFAULT_RATE = 0.0807


def load_cm(data: dict, model_key: str, threshold: str = "optimal") -> np.ndarray | None:
    m = data.get(model_key, {})
    key = "optimal_threshold_metrics" if threshold == "optimal" else "default_threshold"
    raw = m.get(key, {}).get("confusion_matrix")
    return np.array(raw, dtype=np.int64) if raw else None


def get_thresh(data: dict, model_key: str) -> float:
    return data.get(model_key, {}).get("optimal_threshold", 0.5)


def get_auc(data: dict, model_key: str) -> float:
    return data.get(model_key, {}).get("auc_roc", 0.0)


def get_ks(data: dict, model_key: str) -> float:
    return data.get(model_key, {}).get("ks_stat", 0.0)


def annotate_cm(ax, cm: np.ndarray, title: str, thresh: float,
                auc: float, ks: float, label_color: str = "#0D3B36"):
    """Draw a single annotated confusion matrix heatmap."""
    n = cm.sum()
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / row_sums * 100  # row-normalised percentages

    # Background colour map: blue gradient
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")

    # Annotate cells: count + percentage
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_pct[i, j]
            text_color = "white" if pct > 55 else "#1a1a2e"
            ax.text(j, i, f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color=text_color, fontweight="bold")

    # TP / TN / FP / FN labels
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.38, cell_labels[i][j],
                    ha="center", va="center", fontsize=7,
                    color="white" if cm_pct[i, j] > 55 else "#555",
                    fontstyle="italic")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(CLASS_LABELS, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(CLASS_LABELS, fontsize=9, rotation=0)
    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("True label", fontsize=9)
    ax.set_title(title, fontsize=10, color=label_color, fontweight="bold", pad=6)

    # Subtitle with metrics
    ax.text(0.5, -0.28,
            f"Thresh={thresh:.3f} | AUC={auc:.4f} | KS={ks:.4f}",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=7.5, color="#444", fontstyle="italic")


# ─────────────────────────────────────────────────────────────
# Figure 1 — 2×4 grid: all 7 models at optimal threshold
# ─────────────────────────────────────────────────────────────
def fig_all_models(data: dict) -> None:
    n_models = len(MODELS)
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8.5))
    fig.suptitle(
        "Figure 6.5  Confusion Matrices — All Models at Optimal Decision Threshold\n"
        f"(5-fold OOF, n = {POPULATION:,}, Population Default Rate = {DEFAULT_RATE*100:.2f}%)",
        fontsize=12, y=1.01
    )

    axes_flat = axes.flatten()
    for idx, (key, label, color) in enumerate(MODELS):
        ax = axes_flat[idx]
        cm = load_cm(data, key, "optimal")
        if cm is None:
            ax.axis("off")
            ax.set_title(f"{label}\n(no data)", fontsize=9, color="gray")
            continue
        thresh = get_thresh(data, key)
        auc = get_auc(data, key)
        ks = get_ks(data, key)
        annotate_cm(ax, cm, label, thresh, auc, ks, label_color=color)

    # Hide any unused subplot
    for idx in range(n_models, nrows * ncols):
        axes_flat[idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT_DIR, "fig6_5_confusion_matrices_all.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {os.path.basename(out)}")


# ─────────────────────────────────────────────────────────────
# Figure 2 — Single large: Stacking Ensemble champion
# ─────────────────────────────────────────────────────────────
def fig_champion(data: dict) -> None:
    key = "stacking_ensemble"
    cm_opt = load_cm(data, key, "optimal")
    cm_05 = load_cm(data, key, "default")
    thresh = get_thresh(data, key)
    auc = get_auc(data, key)
    ks = get_ks(data, key)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        "Figure 6.6  Stacking Ensemble Champion — Confusion Matrix Comparison\n"
        f"(5-fold OOF, n = {POPULATION:,}, Default Rate = {DEFAULT_RATE*100:.2f}%)",
        fontsize=12, y=1.02
    )

    annotate_cm(axes[0], cm_opt,
                f"At Optimal Threshold (p = {thresh:.3f})",
                thresh, auc, ks, label_color="#0D3B36")
    annotate_cm(axes[1], cm_05,
                "At Default Threshold (p = 0.500)",
                0.500, auc, ks, label_color="#6B2737")

    # Insight annotation
    tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()
    tn_05, fp_05, fn_05, tp_05 = cm_05.ravel()
    recall_opt = tp_opt / (tp_opt + fn_opt) * 100
    recall_05 = tp_05 / (tp_05 + fn_05) * 100

    fig.text(
        0.5, -0.04,
        f"Optimal threshold captures {recall_opt:.1f}% of actual defaults (recall). "
        f"Default threshold captures only {recall_05:.1f}% — "
        "a {:.0f}pp improvement from calibrated threshold selection.".format(recall_opt - recall_05),
        ha="center", fontsize=9, style="italic", color="#333"
    )

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig6_6_confusion_matrix_champion.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {os.path.basename(out)}")


# ─────────────────────────────────────────────────────────────
# Figure 3 — Side-by-side: Champion vs LR baseline
# ─────────────────────────────────────────────────────────────
def fig_compare(data: dict) -> None:
    pairs = [
        ("stacking_ensemble", "Stacking Ensemble (Champion)", "#0D3B36"),
        ("logistic_regression", "Logistic Regression (Baseline)", "#8FA9C4"),
        ("lightgbm", "LightGBM", "#D9534F"),
        ("dnn", "Deep Neural Network", "#B085C9"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        "Figure 6.7  Model Comparison — Confusion Matrices at Optimal Thresholds\n"
        f"(5-fold OOF, n = {POPULATION:,}, Population Default Rate = {DEFAULT_RATE*100:.2f}%)",
        fontsize=12, y=1.02
    )

    for ax, (key, label, color) in zip(axes, pairs):
        cm = load_cm(data, key, "optimal")
        if cm is None:
            ax.axis("off")
            continue
        thresh = get_thresh(data, key)
        auc = get_auc(data, key)
        ks = get_ks(data, key)
        annotate_cm(ax, cm, label, thresh, auc, ks, label_color=color)

        # Recall annotation
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) * 100
        precision = tp / (tp + fp) * 100
        ax.text(0.5, -0.35,
                f"Recall: {recall:.1f}%  |  Precision: {precision:.1f}%",
                ha="center", va="top", transform=ax.transAxes,
                fontsize=8, color=color, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig6_7_confusion_matrix_compare.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {os.path.basename(out)}")


# ─────────────────────────────────────────────────────────────
def main() -> None:
    data = json.load(open(RESULTS_PATH, encoding="utf-8"))
    fig_all_models(data)
    fig_champion(data)
    fig_compare(data)
    print(f"\nAll confusion matrix figures written to {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("fig6_5") or f.startswith("fig6_6") or f.startswith("fig6_7"):
            size_kb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
            print(f"  {f}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
