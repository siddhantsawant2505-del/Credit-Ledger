# Credit Ledger — Model Evaluation & Visualization Suite

This folder contains the complete evaluation module and generated high-resolution visualizations for the 8 benchmarked credit scoring models.

---

## 📁 Folder Structure

```text
models/evaluation/
├── evaluate_and_visualize.py        # Executable evaluation & plotting script
├── evaluation_dashboard.png         # 4-panel executive dashboard
├── model_comparison_benchmark.png   # AUC-ROC vs KS-Statistic vs Optimal F1
├── cv_fold_variance.png             # 5-Fold Stratified CV stability error bars
├── threshold_impact_f1.png          # Decision threshold optimization gains
├── confusion_matrices.png           # Champion (Stacking) vs Baseline (LogReg)
├── roc_curves.png                   # Multi-model ROC comparison curves
└── README.md                        # Documentation & analysis
```

---

## 📊 Benchmark Results Summary

Results from **5-Fold Stratified Cross-Validation (Out-Of-Fold Evaluation)**:

| Rank | Model | Architecture Tier | CV AUC Mean ± Std | OOF AUC-ROC | KS-Statistic | Optimal F1 | Optimal Threshold |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **#1** | **Stacking Ensemble** | Meta-Ensemble | **0.7696 ± 0.0085** | **0.7692** | **0.4025** | 0.2597 | 0.0615 |
| **#2** | **Gradient Boosting** | Ensemble ML | 0.7609 ± 0.0055 | 0.7608 | 0.3965 | **0.2815** | 0.0838 |
| **#3** | **Random Forest** | Ensemble ML | 0.7592 ± 0.0069 | 0.7590 | 0.3876 | 0.2673 | 0.3711 |
| **#4** | **LightGBM** | Ensemble ML | 0.7577 ± 0.0076 | 0.7577 | 0.3841 | 0.2622 | 0.3162 |
| **#5** | **Linear Discriminant (LDA)** | Traditional Baseline | 0.7575 ± 0.0075 | 0.7574 | 0.3887 | 0.2691 | 0.0763 |
| **#6** | **Logistic Regression** | Traditional Baseline | 0.7557 ± 0.0076 | 0.7557 | 0.3921 | 0.2776 | 0.5155 |
| **#7** | **XGBoost** | Ensemble ML | 0.7525 ± 0.0099 | 0.7524 | 0.3781 | 0.2812 | 0.3032 |
| **#8** | **Deep Neural Net (DNN)** | AI-Driven (PyTorch GPU) | 0.7521 ± 0.0096 | 0.7519 | 0.3916 | 0.2770 | 0.5281 |

---

## 📈 Visualizations in this Folder

1. **`evaluation_dashboard.png`** (300 DPI):
   - Executive 4-panel overview displaying AUC/KS comparison, CV stability with error bars, threshold optimization impact on F1, and the precision-recall trade-off scatter.
2. **`model_comparison_benchmark.png`**:
   - Grouped bar chart comparing AUC-ROC, Kolmogorov-Smirnov (KS) statistic, and Optimal F1.
3. **`cv_fold_variance.png`**:
   - Horizontal bar chart showing fold variance and standard deviation error bars.
4. **`threshold_impact_f1.png`**:
   - Comparison of F1 score at default 0.5 threshold vs optimal KS-calibrated threshold.
5. **`confusion_matrices.png`**:
   - Side-by-side confusion matrix heatmaps for Champion (**Stacking Ensemble**) and Baseline (**Logistic Regression**).
6. **`roc_curves.png`**:
   - Multi-model ROC curves demonstrating separation across all 8 models.

---

## 🚀 How to Re-Run

To re-generate all charts and print the report anytime:

```bash
python models/evaluation/evaluate_and_visualize.py
```
