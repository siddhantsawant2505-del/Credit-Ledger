# Loan Default Prediction — Ensemble ML & AI-Driven Credit Scoring

A comparative study of traditional, ensemble, and AI-driven models for predicting loan defaults, based on the reference paper *"Predicting Loan Defaults Using Ensemble Machine Learning And AI-Driven Credit Scoring Models: A Comparative Study"* (Agboola, 2025, IJTMH), implemented on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset with an interactive Streamlit GUI.

## Overview

Financial institutions rely on credit scoring to assess borrower default risk. Traditional models (logistic regression, LDA) assume linear relationships and struggle to capture complex borrower behavior. This project benchmarks three tiers of models — traditional, ensemble, and AI-driven — on a real, multi-table lending dataset, and ships the result as a usable prediction tool rather than just a notebook.

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle) — ~307k applicants across 7 tables:

| Table | Contents |
|---|---|
| `application_train/test.csv` | Main applicant table, target label |
| `bureau.csv` / `bureau_balance.csv` | Prior credit bureau records |
| `previous_application.csv` | Applicant's prior loan applications |
| `POS_CASH_balance.csv` | Point-of-sale / cash loan balances |
| `installments_payments.csv` | Repayment history |
| `credit_card_balance.csv` | Credit card balances |

## Pipeline

```
Raw tables → Feature engineering (aggregation) → Preprocessing & split
    → [Baseline | Ensemble | AI-driven] models → Evaluation & explainability
```

### Models compared

| Tier | Models |
|---|---|
| Traditional baseline | Logistic Regression, Linear Discriminant Analysis |
| Ensemble ML | Random Forest, Gradient Boosting, XGBoost, LightGBM, Stacking |
| AI-driven | Deep Neural Network (DNN), Hybrid model (structured + behavioral features) |

### Evaluation

AUC-ROC, Precision, Recall, F1-Score, KS-statistic, confusion matrix, 5-fold stratified cross-validation, paired t-test/ANOVA for significance, and SHAP for explainability.

## GUI

A Streamlit app with 4 pages, navigated via sidebar:

- **Home** — project summary and navigation
- **Predict & results** — enter applicant data, get a default probability and risk verdict from any trained model
- **Model comparison** — AUC/F1/KS bar charts and ROC curves across all models
- **Explainability** — SHAP feature-importance for the last prediction

## Project structure

```
.
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_ensemble_models.ipynb
│   ├── 06_ai_models.ipynb
│   ├── 07_evaluation.ipynb
│   └── 08_explainability.ipynb
├── app.py
├── pages/
│   ├── 1_Predict_and_Results.py
│   ├── 2_Model_Comparison.py
│   └── 3_Explainability.py
├── utils.py
├── models/            # saved trained models + preprocessing pipeline
├── data/               # raw + processed data (not committed)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd loan-default-prediction
pip install -r requirements.txt

# Download the Home Credit dataset from Kaggle into data/raw/
# Run notebooks 01–08 in order to produce trained models under models/

streamlit run app.py
```

## Results

Benchmarked via 5-Fold Stratified Cross-Validation (Out-Of-Fold Evaluation) with tuned hyperparameters:

| Model | CV AUC Mean ± Std | OOF AUC-ROC | Optimal F1 | KS-Statistic | Status |
|---|---|---|---|---|---|
| **Stacking Ensemble** | **0.7696 ± 0.0085** | **0.7692** | 0.2597 | **0.4025** | Production Champion |
| **Gradient Boosting** | 0.7609 ± 0.0055 | 0.7608 | **0.2815** | 0.3965 | Challenger A |
| **Random Forest** | 0.7592 ± 0.0069 | 0.7590 | 0.2673 | 0.3876 | Validated Benchmark |
| **LightGBM** | 0.7577 ± 0.0076 | 0.7577 | 0.2622 | 0.3841 | High-Speed GBDT |
| **Linear Discriminant (LDA)**| 0.7575 ± 0.0075 | 0.7574 | 0.2691 | 0.3887 | Closed-form Baseline |
| **Logistic Regression** | 0.7557 ± 0.0076 | 0.7557 | 0.2776 | 0.3921 | Regulatory Baseline |
| **XGBoost** | 0.7525 ± 0.0099 | 0.7524 | 0.2812 | 0.3781 | Validated Challenger |
| **Deep Neural Net (DNN)** | 0.7521 ± 0.0096 | 0.7519 | 0.2770 | 0.3916 | PyTorch GPU Accelerated |

## Reference

Agboola, O. K. (2025). Predicting Loan Defaults Using Ensemble Machine Learning And AI-Driven Credit Scoring Models: A Comparative Study. *International Journal of Technology, Management and Humanities, 11*(1). DOI: 10.21590/ijtmh.11.02.03

## Author

Siddhant — Final-year Computer Engineering, St. Francis Institute of Technology (SFIT), Mumbai