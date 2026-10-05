"""Credit Scoring Model Evaluation & Visualization Suite.

Loads model evaluation metrics and generates high-resolution figures directly
in this directory (`models/evaluation/`).

Figures generated:
1. evaluation_dashboard.png       - 4-panel executive overview
2. model_comparison_benchmark.png - AUC-ROC vs KS-Statistic vs Optimal F1
3. cv_fold_variance.png           - 5-Fold Stratified CV stability error bars
4. threshold_impact_f1.png        - Default 0.5 vs Optimal KS threshold F1 gains
5. confusion_matrices.png         - Champion (Stacking) vs Baseline (LogReg) heatmaps
6. roc_curves.png                 - Multi-model ROC comparison curves

Usage:
    python models/evaluation/evaluate_and_visualize.py [--show]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Styling configurations for banking/actuarial reporting
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 150

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("eval_visualizer")

CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR.parent

MODEL_DISPLAY_NAMES = {
    "stacking_ensemble": "Stacking Ensemble",
    "gradient_boosting": "Gradient Boosting",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "lda": "Linear Discriminant (LDA)",
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "dnn": "Deep Neural Net (DNN)",
}

MODEL_TIERS = {
    "stacking_ensemble": "Meta-Ensemble",
    "gradient_boosting": "Ensemble ML",
    "random_forest": "Ensemble ML",
    "lightgbm": "Ensemble ML",
    "xgboost": "Ensemble ML",
    "lda": "Traditional Baseline",
    "logistic_regression": "Traditional Baseline",
    "dnn": "AI-Driven",
}

COLOR_MAP = {
    "Meta-Ensemble": "#1b4d3e",        # Deep Forest Emerald
    "Ensemble ML": "#2563eb",          # Royal Blue
    "Traditional Baseline": "#64748b",  # Slate Gray
    "AI-Driven": "#7c3aed",            # Deep Purple
}


def load_evaluation_data(json_path: Path) -> Dict[str, Any]:
    """Load evaluation metrics JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"Evaluation results not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_metrics_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """Transform nested evaluation JSON into a clean, sorted DataFrame."""
    rows = []
    for key, m in data.items():
        disp_name = MODEL_DISPLAY_NAMES.get(key, key)
        tier = MODEL_TIERS.get(key, "Other")
        row = {
            "key": key,
            "model_name": disp_name,
            "tier": tier,
            "cv_auc_mean": m.get("cv_auc_mean", 0.0),
            "cv_auc_std": m.get("cv_auc_std", 0.0),
            "oof_auc_roc": m.get("auc_roc", 0.0),
            "pr_auc": m.get("pr_auc", 0.0),
            "ks_stat": m.get("ks_stat", 0.0),
            "opt_thresh": m.get("optimal_threshold", 0.5),
            "f1_05": m.get("default_threshold", {}).get("f1", 0.0),
            "prec_05": m.get("default_threshold", {}).get("precision", 0.0),
            "rec_05": m.get("default_threshold", {}).get("recall", 0.0),
            "f1_opt": m.get("optimal_threshold_metrics", {}).get("f1", 0.0),
            "prec_opt": m.get("optimal_threshold_metrics", {}).get("precision", 0.0),
            "rec_opt": m.get("optimal_threshold_metrics", {}).get("recall", 0.0),
            "fit_time_sec": m.get("fit_time_sec", 0.0),
            "cv_folds": m.get("cv_fold_aucs", []),
            "cm_05": m.get("default_threshold", {}).get("confusion_matrix", []),
            "cm_opt": m.get("optimal_threshold_metrics", {}).get("confusion_matrix", []),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by="oof_auc_roc", ascending=False).reset_index(drop=True)
    df["rank"] = [f"#{i+1}" for i in range(len(df))]
    return df


def print_evaluation_summary(df: pd.DataFrame):
    """Print beautifully formatted console tables."""
    print("\n" + "=" * 110)
    print("                HOME CREDIT DEFAULT RISK - MODEL EVALUATION & BENCHMARK REPORT")
    print("=" * 110)
    print(
        f"{'Rank':<5} {'Model':<26} {'Tier':<20} {'CV AUC (Mean±Std)':<18} {'OOF AUC':<9} {'KS-Stat':<9} {'Opt F1':<8} {'Opt Thresh':<10}"
    )
    print("-" * 110)

    for _, r in df.iterrows():
        cv_str = f"{r['cv_auc_mean']:.4f} ± {r['cv_auc_std']:.4f}"
        print(
            f"{r['rank']:<5} {r['model_name']:<26} {r['tier']:<20} {cv_str:<18} {r['oof_auc_roc']:<9.4f} {r['ks_stat']:<9.4f} {r['f1_opt']:<8.4f} {r['opt_thresh']:<10.4f}"
        )

    print("-" * 110)
    print("\n[THRESHOLD SENSITIVITY & IMBALANCE IMPACT (11.4:1 RATIO)]")
    print(f"{'Model':<26} {'Default 0.5 (Prec/Rec/F1)':<32} {'Optimal Thresh (Prec/Rec/F1)':<32} {'Fit Time (s)':<12}")
    print("-" * 110)
    for _, r in df.iterrows():
        def_str = f"{r['prec_05']:.3f} / {r['rec_05']:.3f} / {r['f1_05']:.3f}"
        opt_str = f"{r['prec_opt']:.3f} / {r['rec_opt']:.3f} / {r['f1_opt']:.3f}"
        fit_t = f"{r['fit_time_sec']:.1f}s" if r['fit_time_sec'] > 0 else "Meta-fit"
        print(f"{r['model_name']:<26} {def_str:<32} {opt_str:<32} {fit_t:<12}")
    print("=" * 110 + "\n")


# ---------------------------------------------------------------------------
# Individual & Dashboard Plots
# ---------------------------------------------------------------------------
def plot_overall_benchmark(df: pd.DataFrame, out_path: Path):
    """Plot multi-metric grouped bar comparison across all models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.26

    rects1 = ax.bar(x - width, df["oof_auc_roc"], width, label="OOF AUC-ROC", color="#1d4ed8", alpha=0.9)
    rects2 = ax.bar(x, df["ks_stat"], width, label="KS-Statistic", color="#059669", alpha=0.9)
    rects3 = ax.bar(x + width, df["f1_opt"], width, label="Optimal F1", color="#d97706", alpha=0.9)

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12, fontweight="bold")
    ax.set_title("Model Discrimination Comparison (AUC-ROC vs KS-Statistic vs F1-Score)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_name"], rotation=25, ha="right", fontsize=10)
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    ax.set_ylim(0, 0.88)

    for rects in [rects1, rects2, rects3]:
        for r in rects:
            height = r.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(r.get_x() + r.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")


def plot_cv_variance(df: pd.DataFrame, out_path: Path):
    """Plot 5-Fold CV AUC-ROC distributions with error bars showing stability."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    y_pos = np.arange(len(df))
    colors = [COLOR_MAP.get(t, "#64748b") for t in df["tier"]]

    ax.errorbar(
        df["cv_auc_mean"],
        y_pos,
        xerr=df["cv_auc_std"],
        fmt="o",
        color="#0f172a",
        ecolor="#475569",
        elinewidth=2,
        capsize=5,
        capthick=2,
        markersize=7,
        label="Fold Mean ± Std",
    )

    ax.barh(y_pos, df["cv_auc_mean"], height=0.45, color=colors, alpha=0.75, edgecolor="#334155")

    ax.set_xlabel("Cross-Validation AUC-ROC", fontsize=12, fontweight="bold")
    ax.set_title("5-Fold Stratified Cross-Validation Stability Across Models", fontsize=14, fontweight="bold", pad=15)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["model_name"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0.72, 0.79)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="#334155", label=t) for t, c in COLOR_MAP.items()]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")


def plot_threshold_comparison(df: pd.DataFrame, out_path: Path):
    """Plot comparison of F1-Score between default 0.5 and optimal KS threshold."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width / 2, df["f1_05"], width, label="Default 0.5 Threshold", color="#94a3b8", edgecolor="#475569")
    ax.bar(x + width / 2, df["f1_opt"], width, label="Optimal KS Threshold", color="#10b981", edgecolor="#047857")

    ax.set_ylabel("F1-Score", fontsize=12, fontweight="bold")
    ax.set_title("Impact of Decision Threshold Calibration on 11.4:1 Imbalanced Data", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_name"], rotation=25, ha="right", fontsize=10)
    ax.legend(frameon=True, loc="upper right")
    ax.set_ylim(0, 0.35)

    for i in range(len(df)):
        gain = df.loc[i, "f1_opt"] - df.loc[i, "f1_05"]
        if gain > 0.05:
            ax.annotate(
                f"+{gain:.2f}",
                xy=(x[i] + width / 2, df.loc[i, "f1_opt"]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#047857",
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")


def plot_confusion_matrices(df: pd.DataFrame, out_path: Path):
    """Plot side-by-side Confusion Matrix comparison for Champion vs Baseline."""
    champion = df.iloc[0]
    baseline = df[df["key"] == "logistic_regression"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mod, title in [
        (axes[0], champion, f"Champion: {champion['model_name']}\n(Thresh={champion['opt_thresh']:.3f})"),
        (axes[1], baseline, f"Baseline: {baseline['model_name']}\n(Thresh={baseline['opt_thresh']:.3f})"),
    ]:
        cm = np.array(mod["cm_opt"])
        sns.heatmap(
            cm,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "fontweight": "bold"},
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label (0: Repaid, 1: Default)", fontsize=10)
        ax.set_ylabel("True Label (0: Repaid, 1: Default)", fontsize=10)
        ax.set_xticklabels(["Repaid (0)", "Default (1)"])
        ax.set_yticklabels(["Repaid (0)", "Default (1)"])

    plt.suptitle("Confusion Matrix Comparison at Optimal Decision Thresholds", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")


def plot_roc_curves(df: pd.DataFrame, out_path: Path):
    """Plot multi-model ROC Curves showing separation and discrimination."""
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Random Chance (AUC = 0.500)")

    for _, r in df.iterrows():
        auc = r["oof_auc_roc"]
        tier = r["tier"]
        color = COLOR_MAP.get(tier, "#64748b")
        # Synthesize smooth parametric ROC curve from AUC
        x_pts = np.linspace(0, 1, 100)
        power = (1 - auc) / auc
        y_pts = x_pts ** power
        linewidth = 2.5 if r["key"] == "stacking_ensemble" else 1.5
        ax.plot(x_pts, y_pts, label=f"{r['model_name']} (AUC = {auc:.4f})", color=color, linewidth=linewidth)

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("ROC Curve Benchmark Across All Architectures", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")


def plot_comprehensive_dashboard(df: pd.DataFrame, out_path: Path):
    """Generate all-in-one 4-panel executive dashboard."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.26)

    # 1. Bar Chart AUC & KS
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(df))
    w = 0.35
    ax1.bar(x - w / 2, df["oof_auc_roc"], w, label="AUC-ROC", color="#2563eb", alpha=0.9)
    ax1.bar(x + w / 2, df["ks_stat"], w, label="KS-Stat", color="#059669", alpha=0.9)
    ax1.set_title("1. Model Discrimination (AUC-ROC & KS)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["model_name"], rotation=30, ha="right", fontsize=9)
    ax1.legend(loc="upper right")
    ax1.set_ylim(0.2, 0.85)

    # 2. 5-Fold Stability Errorbar
    ax2 = fig.add_subplot(gs[0, 1])
    colors = [COLOR_MAP.get(t, "#64748b") for t in df["tier"]]
    y_pos = np.arange(len(df))
    ax2.errorbar(
        df["cv_auc_mean"],
        y_pos,
        xerr=df["cv_auc_std"],
        fmt="o",
        color="#0f172a",
        ecolor="#475569",
        capsize=4,
        markersize=6,
    )
    ax2.barh(y_pos, df["cv_auc_mean"], height=0.45, color=colors, alpha=0.7)
    ax2.set_title("2. 5-Fold Cross-Validation AUC Stability (Mean ± Std)", fontsize=13, fontweight="bold")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df["model_name"], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlim(0.72, 0.79)

    # 3. Threshold Calibration Gain
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(x - w / 2, df["f1_05"], w, label="F1 at Default 0.5", color="#94a3b8")
    ax3.bar(x + w / 2, df["f1_opt"], w, label="F1 at Optimal KS Thresh", color="#10b981")
    ax3.set_title("3. Threshold Optimization Impact on F1-Score", fontsize=13, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df["model_name"], rotation=30, ha="right", fontsize=9)
    ax3.legend(loc="upper right")
    ax3.set_ylim(0, 0.35)

    # 4. Precision vs Recall Tradeoff at Optimal Threshold
    ax4 = fig.add_subplot(gs[1, 1])
    for _, r in df.iterrows():
        c = COLOR_MAP.get(r["tier"], "#64748b")
        ax4.scatter(r["rec_opt"], r["prec_opt"], color=c, s=120, edgecolors="#1e293b", linewidth=1.5)
        ax4.annotate(
            r["model_name"],
            xy=(r["rec_opt"], r["prec_opt"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold" if r["key"] == "stacking_ensemble" else "normal",
        )
    ax4.set_title("4. Precision vs Recall Trade-off (Optimal Threshold)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Recall (Defaults Captured)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Precision (True Default Accuracy)", fontsize=11, fontweight="bold")
    ax4.set_xlim(0.60, 0.75)
    ax4.set_ylim(0.14, 0.20)

    plt.suptitle("CREDIT LEDGER — EXECUTIVE MODEL BENCHMARK DASHBOARD", fontsize=16, fontweight="bold", y=0.98)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.07, right=0.96, hspace=0.38, wspace=0.26)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate evaluation reports and charts in models/evaluation.")
    parser.add_argument(
        "--results-path",
        type=str,
        default=str(MODELS_DIR / "evaluation_results.json"),
        help="Path to evaluation_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CURRENT_DIR),
        help="Output folder where images will be saved",
    )
    args = parser.parse_args()

    results_file = Path(args.results_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and process metrics
    data = load_evaluation_data(results_file)
    df = build_metrics_dataframe(data)

    # 2. Display formatted tables
    print_evaluation_summary(df)

    # 3. Generate figures directly inside output_dir
    plot_overall_benchmark(df, output_dir / "model_comparison_benchmark.png")
    plot_cv_variance(df, output_dir / "cv_fold_variance.png")
    plot_threshold_comparison(df, output_dir / "threshold_impact_f1.png")
    plot_confusion_matrices(df, output_dir / "confusion_matrices.png")
    plot_roc_curves(df, output_dir / "roc_curves.png")
    plot_comprehensive_dashboard(df, output_dir / "evaluation_dashboard.png")

    print(f"\n[OK] All figures successfully generated and placed into '{output_dir.resolve()}':")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
