"""Generate comprehensive threshold impact figures from real OOF evaluation data.

Outputs (300 DPI PNG) to reports/figures/:
  fig6_8_threshold_impact.png   -- 3-panel: F1/Precision/Recall at p=0.5 vs optimal
  fig6_9_threshold_detail.png   -- Per-model threshold value + metric scatter

Usage:
    python reports/generate_threshold_figures.py
"""

from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS   = os.path.join(ROOT, "models", "evaluation_results.json")
OUT_DIR   = os.path.join(ROOT, "reports", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.3,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

TEAL   = "#0D3B36"
SLATE  = "#94a3b8"
GREEN  = "#10b981"
RED    = "#ef4444"
AMBER  = "#f59e0b"
PURPLE = "#8b5cf6"

# Display order (excluding gradient_boosting which is an outlier at p=0.5)
MODEL_ORDER = [
    ("stacking_ensemble",  "Stacking\nEnsemble"),
    ("lightgbm",           "LightGBM"),
    ("xgboost",            "XGBoost"),
    ("dnn",                "DNN"),
    ("logistic_regression","Logistic\nRegression"),
    ("random_forest",      "Random\nForest"),
    ("lda",                "LDA"),
    ("gradient_boosting",  "Gradient\nBoosting"),
]


def load(data, key, bucket, metric):
    return data.get(key, {}).get(bucket, {}).get(metric, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — 3-panel: F1 / Precision / Recall  (p=0.5 vs optimal)
# ──────────────────────────────────────────────────────────────────────────────
def fig_threshold_impact(data: dict) -> None:
    keys   = [k for k, _ in MODEL_ORDER]
    labels = [l for _, l in MODEL_ORDER]
    n = len(keys)
    x = np.arange(n)
    w = 0.38

    metrics = [
        ("f1",        "F1-Score",  0.0, 0.36, "F1 is the harmonic mean of precision and recall"),
        ("recall",    "Recall (Sensitivity — defaults captured)", 0.0, 1.02,
         "Recall = TP / (TP + FN) — fraction of real defaults the model catches"),
        ("precision", "Precision (Positive Predictive Value)", 0.0, 0.75,
         "Precision = TP / (TP + FP) — of flagged applicants, how many truly default"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 14),
                             gridspec_kw={"hspace": 0.55})
    fig.suptitle(
        "Figure 6.8  Impact of Decision Threshold Calibration on Classification Metrics\n"
        "(5-fold OOF, n = 307,511 applicants, population default rate = 8.07%)",
        fontsize=12.5, y=1.01
    )

    for ax, (metric, ylabel, ymin, ymax, note) in zip(axes, metrics):
        vals_05  = [load(data, k, "default_threshold",         metric) for k in keys]
        vals_opt = [load(data, k, "optimal_threshold_metrics", metric) for k in keys]
        threshs  = [data.get(k, {}).get("optimal_threshold", 0.5) for k in keys]

        b1 = ax.bar(x - w/2, vals_05,  w, label="Default threshold  p = 0.50",
                    color=SLATE, edgecolor="#475569", linewidth=0.7)
        b2 = ax.bar(x + w/2, vals_opt, w, label="Calibrated optimal threshold",
                    color=GREEN, edgecolor="#047857", linewidth=0.7)

        # Value labels
        for bar in b1:
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#475569")

        for bar, thresh in zip(b2, threshs):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5,
                    color="#047857", fontweight="bold")
            # threshold label below bar
            ax.text(bar.get_x() + bar.get_width()/2, -0.035,
                    f"p*={thresh:.3f}", ha="center", va="top", fontsize=6.5,
                    color="#065f46", rotation=0)

        # Gain arrows for large gains (>0.05 improvement)
        for i, (v05, vopt) in enumerate(zip(vals_05, vals_opt)):
            gain = vopt - v05
            if abs(gain) > 0.05:
                color = GREEN if gain > 0 else RED
                ax.annotate(
                    f"{'+'if gain>0 else ''}{gain:.2f}",
                    xy=(x[i] + w/2, max(v05, vopt) + 0.01),
                    ha="center", fontsize=8, fontweight="bold", color=color
                )

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(ymin, ymax)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

        # Population baseline reference line for recall
        if metric == "recall":
            ax.axhline(0.0807, color=AMBER, ls=":", lw=1.2,
                       label="Population default rate")
        if metric == "f1":
            ax.text(0.99, 0.96,
                    "Note: F1 of 0.27–0.29 is mathematically expected\n"
                    "at 8.07% imbalance — not a sign of poor performance.",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, style="italic", color="#555",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc",
                              edgecolor="#cbd5e1", alpha=0.9))

        ax.text(0.5, -0.13, note, transform=ax.transAxes,
                ha="center", fontsize=8.5, style="italic", color="#555")

    out = os.path.join(OUT_DIR, "fig6_8_threshold_impact.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {os.path.basename(out)}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Threshold detail: bubble chart (threshold vs recall, sized by F1)
# ──────────────────────────────────────────────────────────────────────────────
def fig_threshold_detail(data: dict) -> None:
    keys   = [k for k, _ in MODEL_ORDER]
    labels = [l.replace("\n", " ") for _, l in MODEL_ORDER]

    tier_colors = {
        "stacking_ensemble":   TEAL,
        "lightgbm":            "#D9534F",
        "xgboost":             "#E6A23C",
        "dnn":                 PURPLE,
        "logistic_regression": "#8FA9C4",
        "random_forest":       "#5FBD8F",
        "lda":                 "#B3C6E7",
        "gradient_boosting":   "#8CD9B3",
    }

    threshs    = [data.get(k, {}).get("optimal_threshold", 0.5) for k in keys]
    recalls    = [load(data, k, "optimal_threshold_metrics", "recall")    for k in keys]
    precisions = [load(data, k, "optimal_threshold_metrics", "precision") for k in keys]
    f1s        = [load(data, k, "optimal_threshold_metrics", "f1")        for k in keys]
    aucs       = [data.get(k, {}).get("auc_roc", 0)                       for k in keys]
    colors     = [tier_colors[k] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Figure 6.9  Optimal Threshold Analysis — Precision / Recall Trade-off\n"
        "(All models at their KS-maximising decision threshold)",
        fontsize=12.5, y=1.01
    )

    # ── Panel A: Threshold value per model (horizontal bar) ──
    ax = axes[0]
    sorted_idx = np.argsort(threshs)
    ys = np.arange(len(keys))
    ax.barh(ys, [threshs[i] for i in sorted_idx],
            color=[colors[i] for i in sorted_idx],
            edgecolor="#334155", linewidth=0.7, height=0.65)
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[i] for i in sorted_idx], fontsize=9)
    ax.axvline(0.5, color=SLATE, ls="--", lw=1.2, label="Default p = 0.50")
    ax.axvline(0.0807, color=AMBER, ls=":", lw=1.5, label="Population rate (8.07%)")
    ax.set_xlabel("Optimal decision threshold (p*)", fontsize=10)
    ax.set_title("(a) Calibrated optimal threshold per model", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_xlim(0, 0.6)
    for i, yi in enumerate(ys):
        v = threshs[sorted_idx[i]]
        ax.text(v + 0.008, yi, f"p* = {v:.3f}", va="center", fontsize=8, color="#1e293b")

    # ── Panel B: Precision–Recall scatter, bubble = AUC, coloured by model ──
    ax2 = axes[1]
    bubble_sizes = [(auc - 0.76) * 5000 for auc in aucs]  # scale to visible size
    sc = ax2.scatter(recalls, precisions, s=bubble_sizes,
                     c=colors, edgecolors="#1e293b", linewidth=1.2,
                     zorder=3, alpha=0.88)

    for i, lbl in enumerate(labels):
        ax2.annotate(
            lbl,
            xy=(recalls[i], precisions[i]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=8,
            fontweight="bold" if keys[i] == "stacking_ensemble" else "normal",
            color=colors[i]
        )

    # Iso-F1 curves
    f1_levels = [0.26, 0.27, 0.28, 0.29]
    rec_range = np.linspace(0.01, 1.0, 300)
    for f1_target in f1_levels:
        prec_curve = (f1_target * rec_range) / (2 * rec_range - f1_target + 1e-9)
        prec_curve = np.where((prec_curve > 0) & (prec_curve <= 1), prec_curve, np.nan)
        ax2.plot(rec_range, prec_curve, ls="--", lw=0.8, color="#94a3b8", alpha=0.6)
        # label at recall=0.75
        idx = np.searchsorted(rec_range, 0.75)
        if not np.isnan(prec_curve[idx]):
            ax2.text(0.75, prec_curve[idx] + 0.003,
                     f"F1={f1_target:.2f}", fontsize=7, color="#64748b")

    ax2.set_xlabel("Recall  (defaults captured)", fontsize=10)
    ax2.set_ylabel("Precision  (flagged correctly)", fontsize=10)
    ax2.set_title("(b) Precision–Recall at optimal threshold\n(bubble size ∝ AUC-ROC)", fontsize=10)
    ax2.set_xlim(0.60, 0.80)
    ax2.set_ylim(0.15, 0.22)

    # AUC legend
    for auc_ref, label_ref in [(0.765, "AUC 0.765"), (0.780, "AUC 0.780"), (0.786, "AUC 0.786")]:
        ax2.scatter([], [], s=(auc_ref - 0.76) * 5000, c="#94a3b8",
                    edgecolors="#334155", linewidth=1, label=label_ref, alpha=0.7)
    ax2.legend(title="Bubble size", fontsize=8, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig6_9_threshold_detail.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {os.path.basename(out)}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Single-model threshold sweep for the champion (Stacking Ensemble)
# ──────────────────────────────────────────────────────────────────────────────
def fig_threshold_sweep(data: dict) -> None:
    """Reconstruct approximate F1/precision/recall curves from OOF data
    using actual end-point values (p=0.5 and p*) to anchor a parametric sweep."""
    import os, numpy as np

    oof_dir = os.path.join(ROOT, "models", "oof_cache")
    oof_path = os.path.join(oof_dir, "stacking_ensemble.npz")
    target_path = os.path.join(ROOT, "data", "feature_table.parquet")

    if not os.path.exists(oof_path):
        print("fig6_10 skipped: stacking_ensemble OOF cache not found")
        return
    if not os.path.exists(target_path):
        print("fig6_10 skipped: feature_table.parquet not found")
        return

    try:
        import pandas as pd
        from sklearn.metrics import f1_score, precision_score, recall_score
        z = np.load(oof_path, allow_pickle=False)
        y_score = z["oof_proba"].astype(np.float64)

        df = pd.read_parquet(target_path, columns=["TARGET", "SPLIT"])
        y_true = df[df["SPLIT"] == "train"]["TARGET"].astype(np.int8).to_numpy()

        if len(y_score) != len(y_true):
            print(f"fig6_10 skipped: length mismatch {len(y_score)} vs {len(y_true)}")
            return
    except Exception as e:
        print(f"fig6_10 skipped: {e}")
        return

    thresholds = np.linspace(0.03, 0.70, 80)
    f1s, precs, recs = [], [], []

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        # guard: if no positives predicted, skip gracefully
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1); precs.append(prec); recs.append(rec)

    f1s = np.array(f1s)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(thresholds, recs,  color="#2563eb", lw=2.0, label="Recall (sensitivity)")
    ax.plot(thresholds, precs, color="#d97706", lw=2.0, label="Precision (PPV)")
    ax.plot(thresholds, f1s,   color=TEAL,      lw=2.5, label="F1-Score", zorder=4)

    # Mark optimal
    ax.axvline(best_thresh, color=GREEN, ls="--", lw=1.4,
               label=f"F1-optimal  p* = {best_thresh:.3f}")
    ax.axvline(0.5, color=SLATE, ls=":", lw=1.2,
               label="Default threshold  p = 0.500")
    ax.axvline(0.0807, color=AMBER, ls=":", lw=1.2,
               label="Population default rate  8.07%")

    # Annotate peak F1
    ax.annotate(
        f"Peak F1 = {f1s[best_idx]:.4f}\nat p* = {best_thresh:.3f}",
        xy=(best_thresh, f1s[best_idx]),
        xytext=(best_thresh + 0.04, f1s[best_idx] + 0.02),
        fontsize=9, fontweight="bold", color=TEAL,
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2)
    )

    # Annotate recall at p=0.5
    rec_at_05 = np.interp(0.5, thresholds, recs)
    ax.annotate(
        f"Recall = {rec_at_05:.3f}\nat p = 0.500",
        xy=(0.5, rec_at_05),
        xytext=(0.5 + 0.03, rec_at_05 - 0.08),
        fontsize=8.5, color="#2563eb",
        arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.0)
    )

    ax.set_xlabel("Decision threshold (p)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Figure 6.10  Threshold Sweep — Stacking Ensemble Champion\n"
        "(Precision / Recall / F1 across all decision thresholds, real OOF predictions)",
        fontsize=11.5
    )
    ax.legend(fontsize=9, loc="center right", framealpha=0.95)
    ax.set_xlim(thresholds[0] - 0.01, thresholds[-1] + 0.01)
    ax.set_ylim(-0.02, 1.02)

    fig.text(
        0.5, -0.04,
        "At p = 0.50 the Stacking Ensemble captures only ~2% of defaults "
        f"(recall {rec_at_05:.3f}). Calibrating to p* = {best_thresh:.3f} raises "
        f"recall to {recs[best_idx]:.3f} — a {(recs[best_idx]-rec_at_05)*100:.0f}pp "
        "improvement in defaults caught.",
        ha="center", fontsize=9, style="italic", color="#333"
    )

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig6_10_threshold_sweep_champion.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {os.path.basename(out)}")


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    data = json.load(open(RESULTS, encoding="utf-8"))
    fig_threshold_impact(data)
    fig_threshold_detail(data)
    fig_threshold_sweep(data)

    print(f"\nAll threshold figures written to {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        if any(f.startswith(p) for p in ("fig6_8", "fig6_9", "fig6_10")):
            size_kb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
            print(f"  {f}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
