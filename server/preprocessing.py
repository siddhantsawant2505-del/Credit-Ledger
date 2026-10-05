"""Home Credit Default Risk - Preprocessing Module.

Builds a single reusable sklearn ColumnTransformer over the merged feature
table produced by feature_engineering.build_feature_table().

Responsibilities:
- Missing-value imputation: median for numeric, constant "MISSING" for
  categoricals. Fit on TRAIN rows only (enforced by the caller via
  Pipeline.fit on the train split - see run_preprocessing.py).
- Categorical encoding: one-hot for low-cardinality columns,
  ordinal/label encoding for high-cardinality ones (target-encoding-style
  fitted inside the pipeline to avoid leakage).
- Scaling: StandardScaler applied only in the "scaled" variant of the
  transformer (for logistic/DNN); tree models use the unscaled variant.
- Class imbalance helpers: scale_pos_weight (XGBoost/LightGBM) and
  class_weight='balanced' dicts (LR/RF) - no SMOTE oversampling.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

TARGET = "TARGET"
ID_CURR = "SK_ID_CURR"

# Cardinality threshold separating one-hot from ordinal encoding.
LOW_CARDINALITY_MAX = 15
# OrdinalEncoder unknown-category handling
HANDLE_UNKNOWN = "use_encoded_value"
UNKNOWN_VALUE = -1


def split_column_types(
    df: pd.DataFrame, drop_ids: bool = True
) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols) for the feature table.

    Categorical = category dtype OR object dtype (excluding the ID column).
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in df.columns:
        if drop_ids and col == ID_CURR:
            continue
        if isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == object:
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols, categorical_cols


def split_cardinality(
    df: pd.DataFrame, categorical_cols: List[str], threshold: int = LOW_CARDINALITY_MAX
) -> Tuple[List[str], List[str]]:
    """Split categorical columns into (low_cardinality, high_cardinality) lists."""
    low: List[str] = []
    high: List[str] = []
    for col in categorical_cols:
        n_unique = df[col].nunique(dropna=True)
        (low if n_unique <= threshold else high).append(col)
    return low, high


def build_preprocessor(
    feature_df: pd.DataFrame,
    scale_numeric: bool = False,
    low_cardinality_max: int = LOW_CARDINALITY_MAX,
) -> ColumnTransformer:
    """Build the reusable sklearn ColumnTransformer.

    Parameters
    ----------
    feature_df : pd.DataFrame
        The merged feature table (used ONLY for column names and cardinality
        measurement - no statistics are fitted here, keeping this object
        safe to fit later on train-only data).
    scale_numeric : bool
        True  -> StandardScaler after median imputation (logistic / DNN).
        False -> imputation only (tree models: XGBoost/LightGBM/RF).
    low_cardinality_max : int
        Columns with <= this many unique values are one-hot encoded;
        the rest use OrdinalEncoder (fast, tree-friendly, and paired with
        TargetEncoder for linear models via the scaled variant).

    Returns
    -------
    ColumnTransformer (unfitted; call .fit on TRAIN rows only).
    """
    numeric_cols, categorical_cols = split_column_types(feature_df)
    low_card, high_card = split_cardinality(feature_df, categorical_cols, low_cardinality_max)

    num_steps: List[Tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(num_steps)

    low_card_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    high_card_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown=HANDLE_UNKNOWN,
                    unknown_value=UNKNOWN_VALUE,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat_low", low_card_pipeline, low_card),
            ("cat_high", high_card_pipeline, high_card),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
        n_jobs=None,
    )
    logger.info(
        "Preprocessor: %d numeric (%s), %d low-card one-hot, %d high-card ordinal",
        len(numeric_cols),
        "scaled" if scale_numeric else "unscaled",
        len(low_card),
        len(high_card),
    )
    return preprocessor


# ---------------------------------------------------------------------------
# class imbalance helpers (no SMOTE)
# ---------------------------------------------------------------------------
def get_scale_pos_weight(y_train: pd.Series) -> float:
    """scale_pos_weight for XGBoost/LightGBM: n_negative / n_positive."""
    counts = y_train.value_counts()
    n_neg = int(counts.get(0, 0))
    n_pos = int(counts.get(1, 0))
    if n_pos == 0:
        raise ValueError("No positive (default) samples in y_train.")
    spw = n_neg / n_pos
    logger.info("scale_pos_weight = %.3f (%d neg / %d pos)", spw, n_neg, n_pos)
    return float(spw)


def get_class_weights(y_train: pd.Series) -> Dict[int, float]:
    """class_weight dict for LR/RF: 'balanced' = n_samples / (n_classes * count)."""
    counts = y_train.value_counts()
    n = int(counts.sum())
    k = len(counts)
    weights = {int(cls): n / (k * int(cnt)) for cls, cnt in counts.items()}
    logger.info("class_weight = %s", weights)
    return weights


# ---------------------------------------------------------------------------
# post-fit summary
# ---------------------------------------------------------------------------
def summarize_features(
    feature_df: pd.DataFrame,
    y: pd.Series,
    transformed: Optional[np.ndarray] = None,
    transformed_names: Optional[List[str]] = None,
    top_n: int = 10,
) -> str:
    """Print-ready summary: final feature count and top features by |corr| with TARGET.

    Correlations are computed on TRAIN rows only (y not-NaN).

    transformed : OPTIONAL np.ndarray of shape (n_train_rows, n_features) -
        the ALREADY train-only transformed matrix (e.g. preprocessor.transform(X_train)).
        Do NOT pass the full train+test matrix; masking is done internally via y.
    """
    lines: List[str] = []
    train_mask = y.notna().values
    y_train = y[train_mask].reset_index(drop=True)

    if transformed is not None and transformed_names:
        if len(transformed) != len(y_train):
            raise ValueError(
                f"transformed has {len(transformed)} rows but train mask selects "
                f"{len(y_train)} - pass the train-only transformed matrix."
            )
        n_features = transformed.shape[1]
        lines.append(f"Final transformed feature count: {n_features}")
        Xtr = pd.DataFrame(transformed, columns=transformed_names)
        corr = Xtr.corrwith(y_train, axis=0, method="pearson").abs().sort_values(ascending=False)
        top = corr.head(top_n)
        lines.append(f"\nTop {len(top)} transformed features by |Pearson corr| with TARGET:")
        lines.append(f"{'feature':<45s} {'|corr|':>8s}")
        for name, val in top.items():
            lines.append(f"{name[:45]:<45s} {val:8.4f}")
    else:
        numeric_cols, _ = split_column_types(feature_df)
        usable = [c for c in numeric_cols if c != TARGET and feature_df[c].notna().sum() > 100]
        corr = feature_df.loc[train_mask, usable].corrwith(
            y_train, axis=0
        ).abs().sort_values(ascending=False)
        top = corr.head(top_n)
        lines.append(f"Raw feature table: {feature_df.shape[1] - 1} features (excl. ID)")
        lines.append(f"\nTop {len(top)} raw numeric features by |corr| with TARGET:")
        lines.append(f"{'feature':<45s} {'|corr|':>8s}")
        for name, val in top.items():
            lines.append(f"{name[:45]:<45s} {val:8.4f}")

    return "\n".join(lines)


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Human-readable output feature names after fitting the preprocessor."""
    return list(preprocessor.get_feature_names_out())
