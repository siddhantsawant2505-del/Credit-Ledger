# Model Comparison & Evaluation Results

> **Evaluated at**: 2026-09-30 15:46:59
> **Cross-Validation**: 5-Fold Stratified CV (Out-Of-Fold Evaluation)
> **Training Samples**: 307,511

## Overall Benchmark

| Model | CV AUC Mean ± Std | OOF AUC-ROC | PR-AUC | KS-Statistic | F1 (at 0.5) | F1 (Opt Thresh) | Optimal Thresh |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `logistic_regression` | 0.7694 ± 0.0043 | **0.7693** | 0.2522 | **0.4037** | 0.2758 | **0.2794** | 0.5093 |
| `lda` | 0.7652 ± 0.0041 | **0.7652** | 0.2499 | **0.3982** | 0.1365 | **0.2763** | 0.0742 |
| `xgboost` | 0.7825 ± 0.0034 | **0.7825** | 0.2747 | **0.4255** | 0.3099 | **0.2872** | 0.4252 |
| `lightgbm` | 0.7838 ± 0.0037 | **0.7838** | 0.2750 | **0.4272** | 0.2960 | **0.2817** | 0.4526 |
| `random_forest` | 0.7648 ± 0.0041 | **0.7648** | 0.2483 | **0.3982** | 0.3080 | **0.2739** | 0.3801 |
| `gradient_boosting` | 0.7699 ± 0.0037 | **0.7699** | 0.2582 | **0.4042** | 0.0514 | **0.2713** | 0.0745 |
| `dnn` | 0.7724 ± 0.0046 | **0.7724** | 0.2648 | **0.4127** | 0.2779 | **0.2777** | 0.4990 |
| `stacking_ensemble` | 0.7860 ± 0.0034 | **0.7860** | 0.2779 | **0.4305** | 0.0377 | **0.2881** | 0.0770 |

## Key Insights & Discussion

- **Class Imbalance Impact**: Notice the stark difference between default 0.5 threshold F1 and optimal KS threshold F1. Due to the 11.4:1 class imbalance, threshold tuning is essential for non-weighted estimators.
- **Ensemble Superiority**: Gradient Boosted trees (LightGBM / XGBoost) and the Stacking Ensemble achieve the highest AUC and KS discrimination.
- **Artifacts Saved**: All models are persisted in `models/` for real-time inference via the API server.
