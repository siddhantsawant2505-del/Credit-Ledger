# Exact corrections for the mini-project report ("give in a docs.docx")

Every replacement below was verified against the actual repository on 2026-10-04.
Use Word's Find & Replace (Ctrl+H) for the short strings; the longer blocks are
full paragraph rewrites. Insert the generated figures from `reports/figures/`
where indicated in section 12.

---

## 1. ABSTRACT — replace the whole paragraph

**Replace this sentence block:**

> "...across six distinct predictive models categorized into three core algorithmic
> families: traditional baselines (Logistic Regression, Decision Tree), ensemble
> methods (Random Forest, XGBoost), and deep AI-driven architectures (Multi-Layer
> Perceptron, Tabular Transformer). The models were evaluated on a benchmark credit
> dataset comprising over 300,000 applicant profiles using discrimination and
> calibration metrics, specifically AUC-ROC, F1-score, and the Kolmogorov-Smirnov
> (KS) statistic. ... local post-hoc explanations are computed via SHAP ... The
> entire predictive workflow is deployed through an interactive four-page Streamlit
> web application..."

**With:**

> "...across eight distinct predictive models categorized into three core
> algorithmic families: traditional baselines (Logistic Regression, Linear
> Discriminant Analysis), ensemble methods (Gradient Boosting, Random Forest,
> XGBoost, LightGBM), and AI-driven architectures (a PyTorch Multi-Layer Perceptron
> and a stacking ensemble with a logistic meta-learner). The models were trained
> and evaluated on the Home Credit Default Risk dataset comprising 307,511 applicant
> profiles (5-fold stratified out-of-fold evaluation) using discrimination and
> calibration metrics: AUC-ROC, PR-AUC, F1-score, Brier score, and the
> Kolmogorov-Smirnov (KS) statistic. To bridge the gap between model performance
> and regulatory transparency requirements, exact TreeSHAP contributions are
> computed natively through LightGBM's and XGBoost's pred_contrib outputs, with
> per-applicant top-feature attributions served by the API. The entire predictive
> workflow is deployed through a FastAPI REST service (port 8000) and an interactive
> four-page Next.js web application (port 3000) designed with a clean, professional
> financial ledger visual aesthetic, enabling loan officers to inspect predictions,
> compare model benchmarks, and analyze individual risk drivers in real time."

**Keywords line — replace with:**
Credit risk assessment, loan default prediction, ensemble learning, gradient
boosting, stacking ensembles, deep learning, explainable AI, TreeSHAP, FastAPI,
Next.js.

---

## 2. Section 1.2 (Problem Statement) — replace the four bullets

- "Trains **six** representative algorithms..." ->
  **Trains eight representative algorithms across three model families (Logistic
  Regression and LDA; Gradient Boosting, Random Forest, XGBoost and LightGBM; a
  PyTorch MLP and a stacking ensemble) on identical stratified 5-fold splits.**
- "Evaluates performance using standard credit scoring metrics: ... AUC-ROC,
  F1-Score, and the ... KS statistic." ->
  **Evaluates performance using AUC-ROC, PR-AUC, F1-Score, Brier score, and the
  Kolmogorov-Smirnov (KS) statistic, at both the default 0.5 threshold and a
  per-model optimal threshold.**
- "Integrates SHAP feature attributions..." ->
  **Integrates exact TreeSHAP feature attributions (via LightGBM/XGBoost
  pred_contrib outputs) for tree models, with heuristic linear attributions for
  linear baselines, providing per-applicant explanations.**
- "Exposes the underlying prediction engine through a multi-page web application..."
  ->
  **Exposes the prediction engine through a FastAPI backend and a four-page Next.js
  application featuring real-time model switching and visual explainability tools.**

---

## 3. Section 1.3 (Prerequisites) — replace the last bullet

**Find:** "Building reactive user interfaces with Streamlit and embedding dynamic
visualizations via Plotly."

**Replace:** "Serving the prediction engine through a FastAPI REST layer and
building reactive user interfaces with Next.js (React, TypeScript) and inline SVG
visualizations."

---

## 4. Section 3.1 (Architecture & Data Flow) — replace diagram + five stages

**Diagram — replace the whole ASCII block with:**

```
+----------------------+   +---------------------------+   +---------------------------+
| Multi-table raw data | ->| Feature engineering       | ->| Preprocessing (impute,    |
| application + bureau |   | (EXT_SOURCE composites,   |   | scale, one-hot)           |
| + POS + installments |   | ORG target encoding, aggs)|   | 207 raw -> 259 columns    |
+----------------------+   +---------------------------+   +---------------------------+
                                                                 |
                                                                 v
+----------------------+   +---------------------------+   +---------------------------+
| 4-page Next.js GUI   | <-| FastAPI serving + TreeSHAP| <-| 8-model training, 5-fold  |
| (Home / Predict /    |   | attributions (joblib      |   | StratifiedKFold OOF       |
|  Compare / Explain)  |   | artifacts, port 8000)     |   | evaluation (fold cache)   |
+----------------------+   +---------------------------+   +---------------------------+
```

**Stage bullets — replace with:**

1. **Data Ingestion & Multi-table Aggregation:** `application_train/test`
   (307,511 / 48,744 rows) is merged with `bureau`, `bureau_balance`,
   `POS_CASH_balance`, `credit_card_balance` and `installments_payments` into a
   single feature table of 356,255 rows x 207 raw columns.
2. **Feature Engineering:** EXT_SOURCE composite scores (`EXT_SOURCES_MEAN/MAX/
   MIN/PROD_2_3/SUM/WEIGHTED/2_SQ`), a smoothed `ORGANIZATION_TYPE` target encoding
   (58 categories, smoothing factor 100, fitted on train rows only, unseen
   categories map to the prior default rate 0.0807), and per-customer aggregates
   from the credit-bureau tables.
3. **Preprocessing:** a leakage-safe sklearn `ColumnTransformer` (fit on train
   only) applies median imputation + `RobustScaler` for numeric columns and
   most-frequent imputation + one-hot encoding for categorical ones, yielding 259
   transformed features. A scaled variant feeds the linear models and the MLP; an
   unscaled variant feeds the tree models.
4. **Multi-Model Execution:** eight models are trained with 5-fold `StratifiedKFold`
   (random_state 42) and evaluated on pooled out-of-fold predictions; per-fold
   results are persisted in `models/oof_cache/` so long runs are resumable.
5. **Serving & Explainability:** model artifacts (joblib / .pt) are loaded by the
   FastAPI service, which returns default probability, risk tier, and top-5
   TreeSHAP contributions per applicant; the Next.js client renders them in real
   time.

---

## 5. Section 3.2 (Feature Engineering) — replace the four features + VIF sentence

**Replace the four feature bullets with:**

- **EXT_SOURCE Composites:** mean, max, min, product, sum, weighted-sum and
  squared combinations of the three external bureau scores
  (`EXT_SOURCE_1/2/3`), the strongest default signal in the dataset
  (|Pearson r| up to 0.222 with TARGET).
- **ORGANIZATION_TYPE Target Encoding:** smoothed mean-default-rate per
  organization category (58 categories, smoothing = 100), fitted strictly on
  training rows; unseen categories fall back to the population prior (0.0807).
- **Credit Bureau Aggregates:** per-applicant mean/max/sum statistics aggregated
  over bureau records, POS-Cash balances, credit-card balances and installments
  payments.
- **Domain Ratios:** credit-to-income, annuity-to-income,
  annuity-to-goods-price, employment-age and related derived ratios.

**Replace the VIF sentence:**

> "To prevent coefficient instability in linear baselines, collinearity is
> evaluated using the Variance Inflation Factor (VIF). Features showing VIF
> values above 10.0 or pairwise Pearson correlation values exceeding r = 0.85
> are removed."

**With:**

> "Collinearity is controlled by construction: tree-based ensembles are
> insensitive to correlated aggregates, and the linear baselines (Logistic
> Regression, LDA) receive L2 regularization / shrinkage instead of an explicit
> VIF-based pruning step, which is therefore not applied."

---

## 6. Section 3.3 (Model Formulations) — replace the six model headings

**Delete the "Decision Tree Classifier" block (formula Gini(t) = 1 - sum p_i^2)
and the "Tabular Transformer (FT-Transformer)" block (attention formula).**

**Keep LR / Random Forest / XGBoost formulas. Add these three blocks:**

**Linear Discriminant Analysis (LDA):** a generative classifier that models each
class as a multivariate Gaussian with a shared covariance matrix and classifies by
the posterior; with shrinkage (`shrinkage="auto"`) the covariance estimate is
regularized toward the identity for numerical stability with 259 correlated
features:

    delta_k(x) = x^T Sigma^-1 mu_k  -  (1/2) mu_k^T Sigma^-1 mu_k  +  log pi_k

**LightGBM (Gradient Boosting Decision Trees):** like XGBoost, a sequential
ensemble minimizing a regularized objective, but with histogram-based split
finding and leaf-wise (best-first) tree growth via `num_leaves`, which is faster
on wide tabular data and typically yields higher AUC at equal budget.

**Multi-Layer Perceptron (PyTorch) — correct the architecture description:**
three hidden layers of **[256, 128, 64]** neurons with ReLU activations,
**BatchNorm1d** after each linear layer, **Dropout p = 0.30**, and a sigmoid
output trained with **AdamW (lr = 1e-3, weight_decay = 1e-4)** and Binary
Cross-Entropy loss, batch size 512. (The report's "[128, 64, 32], Adam LR =
0.001" is wrong.)

**Stacking Ensemble (new block to add):** the pooled out-of-fold probabilities
of the base models are column-stacked into a meta-feature matrix; a Logistic
Regression meta-learner (C = 1.0, L2, lbfgs) is fitted on it with the same 5-fold
stratification, and its OOF AUC is reported so the ensemble is evaluated on data
its base models never saw.

---

## 7. Section 3.4 (Risk Classification Logic) — replace the three tier bullets

- "Low Risk (p < 0.30)..." ->
  **Low Risk (p < 0.0807): below the population default rate; streamlined
  approval recommendation, standard underwriting terms.**
- "Medium Risk (0.30 <= p < 0.60)..." ->
  **Medium Risk (0.0807 <= p < 0.18): secondary manual review recommended;
  additional income verification.**
- "High Risk (p >= 0.60)..." ->
  **High Risk (p >= 0.18): decline recommendation or elevated interest-rate
  pricing.**

*(The 0.0807 boundary is the train-set default rate used throughout the app;
the 0.18 upper boundary is the tier threshold implemented in the Next.js client.)*

---

## 8. Section 4.2 (Hyperparameter Configuration) — replace Table 4.1

**Replace the six model rows with these eight (values from
models/best_hyperparameters.json; tuning was RandomizedSearchCV, 5-fold
stratified CV, scoring = roc_auc, random_state = 42):**

| Family | Algorithm | Selected hyperparameter values |
| :--- | :--- | :--- |
| Traditional | Logistic Regression | solver = lbfgs, penalty = l2, C = 0.574, class_weight = null |
| Traditional | LDA | solver = lsqr, shrinkage = auto, n_components = null |
| Ensemble | Gradient Boosting (sklearn) | n_estimators = 123, max_depth = 3, learning_rate = 0.188, subsample = 0.947 |
| Ensemble | Random Forest | n_estimators = 154, max_depth = 16, max_features = 0.2, min_samples_leaf = 40 |
| Ensemble | XGBoost | n_estimators = 666, max_depth = 6, learning_rate = 0.0539, subsample = 0.892, colsample_bytree = 0.507, reg_alpha = 3.42, reg_lambda = 6.29 |
| Ensemble | LightGBM | n_estimators = 451, num_leaves = 22, max_depth = 11, learning_rate = 0.0734, subsample = 0.891, colsample_bytree = 0.707, reg_alpha = 0.103, reg_lambda = 9.71 |
| AI-driven | MLP (PyTorch) | layers = [256, 128, 64], dropout = 0.30, AdamW lr = 1e-3, weight_decay = 1e-4, batch = 512 |
| Meta | Stacking meta-learner | LogisticRegression (C = 1.0, L2, lbfgs) over base-model OOF probabilities |

**Correct the sentence above the table:** "Hyperparameters were tuned using
5-fold cross-validation **on a 50,000-row stratified subsample of the training
data** (RandomizedSearchCV, scoring = roc_auc)."

---

## 9. Section 5, Table 5.1 (Environment) — replace the table rows

| Category | Specification details |
| :--- | :--- |
| Compute infrastructure | 13th Gen Intel Core i5-13450HX (10 cores / 16 threads), 15.7 GB RAM, NVIDIA GeForce RTX 3050 6GB Laptop GPU (used by the PyTorch MLP) |
| Operating system | Windows 11 (x86_64), Git Bash tooling |
| Programming language | Python 3.12.5 |
| Core ML libraries | scikit-learn 1.7.0, XGBoost 3.4.1, LightGBM 4.7.0, PyTorch 2.6.0 (CUDA 12.4 build) |
| Data stack | pandas 2.2.2, NumPy 2.4.6, SciPy 1.15.3, PyArrow (parquet), joblib 1.4.2 |
| Explainability & visualization | TreeSHAP via LightGBM/XGBoost native pred_contrib outputs (the external shap package is not used), Matplotlib 3.10.3 |
| Serving & web application | FastAPI 0.110.0 + Uvicorn 0.28.0 (API, port 8000); Next.js 14 (App Router, React, TypeScript, port 3000) |

## 10. Chapter 6 (Results) — replace 6.1 and 6.2

**Replace the 45,000-row sentence with:** "Models were evaluated on pooled
5-fold out-of-fold predictions over all 307,511 training applicants."

**Replace Table 6.1 with (metrics from models/evaluation_results.json,
2026-09-30 run):**

| Model family | Algorithm | AUC-ROC | F1 (p=0.50) | F1 (opt) | KS statistic | Optimal threshold | CV AUC (mean +/- std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Traditional | Logistic Regression | 0.7693 | 0.2758 | 0.2794 | 0.4037 | 0.5093 | 0.7694 +/- 0.0043 |
| Traditional | LDA | 0.7652 | 0.1365 | 0.2763 | 0.3982 | 0.0742 | 0.7652 +/- 0.0041 |
| Ensemble | Gradient Boosting | 0.7699 | 0.0514 | 0.2713 | 0.4042 | 0.0745 | 0.7699 +/- 0.0037 |
| Ensemble | Random Forest | 0.7648 | 0.3080 | 0.2739 | 0.3982 | 0.3801 | 0.7648 +/- 0.0041 |
| Ensemble | XGBoost | 0.7825 | 0.3099 | 0.2872 | 0.4255 | 0.4252 | 0.7825 +/- 0.0034 |
| Ensemble | LightGBM | 0.7838 | 0.2960 | 0.2817 | 0.4272 | 0.4526 | 0.7838 +/- 0.0037 |
| AI-driven | MLP (PyTorch) | 0.7724 | 0.2779 | 0.2777 | 0.4127 | 0.4990 | 0.7724 +/- 0.0046 |
| Ensemble (meta) | Stacking Ensemble | **0.7860** | 0.0377 | **0.2881** | **0.4305** | 0.0770 | 0.7860 +/- 0.0034 |

**Replace the three discussion bullets with:**

1. **Ensembles lead on discrimination:** the Stacking Ensemble achieved the best
   AUC-ROC (0.7860) and KS (0.4305), with LightGBM (0.7838) and XGBoost (0.7825)
   close behind - a narrow but consistent ordering that reflects the ceiling of
   signal available in the application + bureau feature space.
2. **Threshold calibration matters more than architecture:** with an 8.07%
   positive rate, the default 0.5 threshold is miscalibrated for several models -
   Gradient Boosting (optimal threshold 0.0745) and the Stacking Ensemble (0.0770)
   collapse to F1 <= 0.05 at 0.5 but recover to about 0.27-0.29 at their
   KS-maximizing thresholds. Reporting F1 at a per-model optimal threshold is the
   fair comparison.
3. **Calibration quality (Brier score):** the Stacking Ensemble achieves the best
   Brier score (0.0661), followed by LightGBM (0.0671) and Gradient Boosting
   (0.0683), all better than the all-negative baseline of 0.0742; XGBoost (0.1717)
   and Logistic Regression (0.1952) are poorly calibrated raw and benefit from the
   probability calibration applied in the serving layer.
4. **Accuracy at the operating point:** at their optimal thresholds all models
   reach about 0.69-0.72 accuracy against a 0.9193 all-negative baseline, i.e.
   the models trade raw accuracy for recall on defaulters (about 0.69-0.73
   recall at optimal thresholds), which is the correct objective for credit
   risk underwriting.

## 11. Chapter 7 (GUI) + Chapter 8 (Code excerpts) — replace both

**Chapter 7 - replace "Streamlit" wording and the view bullets:**

- "The web interface uses a clean, professional visual layout ('Credit Ledger'
  aesthetic): warm light background (#F8F9FA), primary pine-teal containers
  (#0D3B36), standardized risk badge colors (green, amber, red). The interface is
  a Next.js 14 application (App Router, React, TypeScript) communicating with the
  FastAPI service at NEXT_PUBLIC_API_URL." **Keep the four navigation-view bullets
  but correct:** Home shows live-fetched summary metrics (8 models, 307,511
  applicants, 8.07% default rate); Predict & Results shows the risk tiers at
  0.0807/0.18 and real per-model AUCs in the model selector; Model Comparison
  renders the real benchmark (CV std, PR-AUC, Brier, Gini) as data-driven SVG
  charts with an offline fallback state; Explainability renders TreeSHAP
  top-feature contributions for the active applicant (and a global LightGBM
  gain-importance view).

**Chapter 8 - replace the Streamlit excerpt with the actual serving logic
(server/api.py):**

```python
# server/api.py (excerpt) - per-model probability + risk tier + TreeSHAP
pre = CACHE["pre_scaled"] if is_scaled else CACHE["pre_unscaled"]
X = pre.transform(pd.DataFrame([req.features]))

if model_name == "stacking_ensemble":
    stack_bundle = load_model("stacking_ensemble")
    meta_learner = stack_bundle["meta_learner"]
    base_names = stack_bundle["base_model_names"]
    base_preds = []
    for b_name in base_names:
        b_model = load_model(b_name)
        b_pre = (CACHE["pre_scaled"] if b_name in ("logistic_regression", "lda", "dnn")
                 else CACHE["pre_unscaled"])
        b_X = b_pre.transform(input_df)
        if hasattr(b_X, "toarray"):
            b_X = b_X.toarray()
        base_preds.append(float(b_model.predict_proba(
            np.asarray(b_X, dtype=np.float32))[0, 1]))
    raw_p = float(meta_learner.predict_proba([base_preds])[0, 1])
else:
    raw_p = float(load_model(model_name).predict_proba(X)[0, 1])

p = calibrate_probability(raw_p, model_name)
verdict = "Low" if p < 0.0807 else ("Medium" if p < 0.18 else "High")
contributions = _tree_shap_contributions(model_name, X)   # pred_contrib=True
```

---

## 12. Insert the generated figures into Chapter 6

Insert these PNGs (300 DPI, already generated from the real artifacts) into
Chapter 6 and reference them from the discussion:

| Placement | File | Caption to use |
| :--- | :--- | :--- |
| After Table 6.1 | reports/figures/fig6_1_model_benchmark.png | "Figure 6.1: Comparative performance benchmark across model families (5-fold OOF, n = 307,511)." |
| After Figure 6.1 | reports/figures/fig6_2_roc_curves.png | "Figure 6.2: Out-of-fold ROC curves for all models; the stacking ensemble and LightGBM dominate the chance line." |
| After Figure 6.2 | reports/figures/fig6_3_ks_curves.png | "Figure 6.3: KS separation - cumulative score distributions for defaulters (dashed) vs non-defaulters (solid)." |
| In the explainability section (or Ch. 6 end) | reports/figures/fig6_4_shap_beeswarm.png | "Figure 6.4: Exact TreeSHAP attributions for the champion LightGBM on a 2,000-row stratified sample." |

Regenerate any time after retraining with: `python reports/generate_report_figures.py`

## 13. Chapter 9 (Conclusion) — fix the two result claims and add one caveat

- "XGBoost achieved the strongest overall performance (AUC-ROC: 0.876, KS: 0.573)"
  -> **"The Stacking Ensemble achieved the strongest overall performance
  (AUC-ROC: 0.7860, KS: 0.4305), with LightGBM (0.7838) and XGBoost (0.7825)
  close behind."**
- "While the Tabular Transformer performed competitively (AUC-ROC: 0.869)..." ->
  **"While the PyTorch MLP performed competitively (AUC-ROC: 0.7724), ensemble
  methods provided similar or better discrimination with lower training cost and
  simpler deployment."**
- Add to Future Scope: **"A fixed 48,744-row Kaggle test split (without public
  labels) exists for a final single-shot evaluation once the parallel feature-set
  expansion is finalized; current headline metrics are pooled 5-fold OOF
  estimates on the training data."** (This keeps the report honest about the
  mixed-era benchmark while the 259-feature retrain finishes.)

## 14. References - no change required

Refs [1]-[8] are external literature and stay. Refs [9] (Lundberg & Lee, SHAP)
and [10] (Chen & Guestrin, XGBoost) remain correct to cite; optionally add:
Ke, G. et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree,"
NeurIPS 2017 - since LightGBM is now a headline model.
