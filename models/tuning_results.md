# Hyperparameter Tuning Results

- Search: RandomizedSearchCV, 5-fold stratified CV, scoring=roc_auc
- random_state=42 everywhere; final models trained separately

| Model | Best CV AUC | n_iter | Elapsed (s) | Best Params |
| :--- | :--- | :--- | :--- | :--- |
| gradient_boosting | 0.7641 | 6 | 808.2 | `{"learning_rate": 0.18832501471299254, "max_depth": 3, "n_estimators": 123, "subsample"...` |
| logistic_regression | 0.7606 | 8 | 215.6 | `{"C": 0.5741157902710026, "class_weight": null, "penalty": "l2", "solver": "lbfgs"}` |
| lda | 0.7602 | 12 | 6.5 | `{"solver": "lsqr", "shrinkage": "auto", "n_components": null}` |
| lightgbm | 0.7588 | 8 | 151.2 | `{"colsample_bytree": 0.7066620013846402, "learning_rate": 0.07340463611641415, "max_dep...` |
| xgboost | 0.7559 | 8 | 281.8 | `{"colsample_bytree": 0.5071848134990964, "learning_rate": 0.05386982686820831, "max_dep...` |
| random_forest | 0.7553 | 5 | 225.3 | `{"max_depth": 16, "max_features": 0.2, "min_samples_leaf": 40, "n_estimators": 154}` |

## Notes

- Tuning data: see `data_size` in best_params JSON.
- LDA / GradientBoosting: no class-weight hook; handle imbalance via
  threshold tuning at training time.
- Stacking meta-learner and DNN tuning handled separately.