"""Home Credit Default Risk - Feature Engineering Module.

Aggregates the satellite tables (bureau, bureau_balance, previous_application,
installments_payments, POS_CASH_balance, credit_card_balance) to SK_ID_CURR
level and derives application-level financial ratios, producing a single
merged feature table ready for preprocessing.

Design rules:
- All aggregates are computed per SK_ID_CURR (one row per applicant).
- Divisions are guarded (zero/inf -> NaN) so downstream imputation handles them.
- Merges always left-join from the application side, so the orphan child keys
  present in the real data (4-11% of SK_ID_PREV etc.) never drop applicant rows.
- train and test are featurized together (TARGET NaN for test rows) so both
  splits end up with identical feature columns.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "TARGET"
ID_CURR = "SK_ID_CURR"
ID_BUREAU = "SK_ID_BUREAU"

# ORGANIZATION_TYPE target encoding: the raw 58-category column is unusable
# for linear models (one-hot explodes dimensionality; the ordinal pipeline
# imposes a meaningless order). A smoothed, train-only-fitted mean of TARGET
# per category gives every model direct access to the strongest categorical
# signal. Smoothing shrinks small categories toward the global prior.
ORG_TYPE_COL = "ORGANIZATION_TYPE"
ORG_TYPE_ENCODED = "ORGANIZATION_TYPE_TE"
ORG_TE_SMOOTHING = 100.0


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Division that yields NaN instead of inf on zero/NaN denominators."""
    out = numer / denom.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _agg_by_curr(child: pd.DataFrame, specs: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """Group a child table by SK_ID_CURR; specs are (out_name, source_col, aggfunc).

    Returns a DataFrame with SK_ID_CURR as the first column.
    """
    agg_kwargs = {out: (src, func) for out, src, func in specs}
    out = child.groupby(ID_CURR, sort=False).agg(**agg_kwargs).reset_index()
    return out


# ----------------------------------------------------------------------------
# bureau + bureau_balance
# ----------------------------------------------------------------------------
def featurize_bureau(
    bureau: pd.DataFrame, bureau_balance: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Prior bureau loans: counts, active/closed ratio, overdue stats, credit sums.

    bureau_balance is aggregated through its SK_ID_BUREAU link into bureau
    first, so every balance feature collapses to SK_ID_CURR cleanly.
    """
    b = bureau.copy()
    status_flag_cols: List[str] = []

    if bureau_balance is not None:
        bb = bureau_balance.copy()
        # STATUS: C=closed, X=unknown, 0=no DPD, 1-5 = months-past-due buckets.
        # One-hot the statuses at bureau-loan level, then sum them up per applicant.
        # astype(object) first: the loader stores STATUS as category, and fillna
        # with a value outside the existing categories would raise.
        status_series = bb["STATUS"].astype(object)
        status_flags = pd.get_dummies(status_series.fillna("MISSING"), prefix="BB_STATUS", dtype=np.int8)
        status_flag_cols = list(status_flags.columns)
        bb_aug = pd.concat([bb[[ID_BUREAU, "MONTHS_BALANCE"]], status_flags], axis=1)
        bb_by_bureau = bb_aug.groupby(ID_BUREAU, sort=False).agg(
            BB_MONTHS_MIN=("MONTHS_BALANCE", "min"),
            BB_MONTHS_MAX=("MONTHS_BALANCE", "max"),
            **{c: (c, "sum") for c in status_flag_cols},
        )
        b = b.merge(bb_by_bureau.reset_index(), on=ID_BUREAU, how="left")

    b["_active"] = (b["CREDIT_ACTIVE"] == "Active").astype(np.int8)
    b["_closed"] = (b["CREDIT_ACTIVE"] == "Closed").astype(np.int8)

    specs: List[Tuple[str, str, str]] = [
        ("BUREAU_LOAN_COUNT", ID_CURR, "count"),
        ("BUREAU_ACTIVE_COUNT", "_active", "sum"),
        ("BUREAU_CLOSED_COUNT", "_closed", "sum"),
        ("BUREAU_DAYS_CREDIT_MEAN", "DAYS_CREDIT", "mean"),
        ("BUREAU_DAYS_CREDIT_MIN", "DAYS_CREDIT", "min"),
        ("BUREAU_CREDIT_DAY_OVERDUE_MAX", "CREDIT_DAY_OVERDUE", "max"),
        ("BUREAU_CREDIT_DAY_OVERDUE_MEAN", "CREDIT_DAY_OVERDUE", "mean"),
        ("BUREAU_AMT_CREDIT_SUM_TOTAL", "AMT_CREDIT_SUM", "sum"),
        ("BUREAU_AMT_CREDIT_SUM_MEAN", "AMT_CREDIT_SUM", "mean"),
        ("BUREAU_AMT_CREDIT_SUM_DEBT_TOTAL", "AMT_CREDIT_SUM_DEBT", "sum"),
        ("BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX", "AMT_CREDIT_MAX_OVERDUE", "max"),
    ]
    if "BB_MONTHS_MIN" in b.columns:
        specs += [
            ("BUREAU_BAL_MONTHS_MIN", "BB_MONTHS_MIN", "min"),
            ("BUREAU_BAL_MONTHS_MAX", "BB_MONTHS_MAX", "max"),
        ]
    specs += [(f"BUREAU_{c}_SUM", c, "sum") for c in status_flag_cols]

    out = _agg_by_curr(b, specs)

    # guarded ratios
    out["BUREAU_ACTIVE_RATIO"] = _safe_div(out["BUREAU_ACTIVE_COUNT"], out["BUREAU_LOAN_COUNT"])
    out["BUREAU_DEBT_CREDIT_RATIO"] = _safe_div(
        out["BUREAU_AMT_CREDIT_SUM_DEBT_TOTAL"], out["BUREAU_AMT_CREDIT_SUM_TOTAL"]
    )
    return out


# ----------------------------------------------------------------------------
# previous_application
# ----------------------------------------------------------------------------
def featurize_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """Prior applications: count, approval rate, requested vs granted amounts."""
    p = prev.copy()
    p["_approved"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(np.int8)
    p["_refused"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(np.int8)
    p["_canceled"] = (p["NAME_CONTRACT_STATUS"] == "Canceled").astype(np.int8)

    specs = [
        ("PREV_APP_COUNT", ID_CURR, "count"),
        ("PREV_APPROVED_COUNT", "_approved", "sum"),
        ("PREV_REFUSED_COUNT", "_refused", "sum"),
        ("PREV_CANCELED_COUNT", "_canceled", "sum"),
        ("PREV_AMT_APPLICATION_MEAN", "AMT_APPLICATION", "mean"),
        ("PREV_AMT_APPLICATION_MAX", "AMT_APPLICATION", "max"),
        ("PREV_AMT_CREDIT_MEAN", "AMT_CREDIT", "mean"),
    ]
    out = _agg_by_curr(p, specs)
    out["PREV_APPROVAL_RATE"] = _safe_div(out["PREV_APPROVED_COUNT"], out["PREV_APP_COUNT"])
    out["PREV_REFUSAL_RATE"] = _safe_div(out["PREV_REFUSED_COUNT"], out["PREV_APP_COUNT"])
    out["PREV_GRANTED_VS_REQUESTED"] = _safe_div(
        out["PREV_AMT_CREDIT_MEAN"], out["PREV_AMT_APPLICATION_MEAN"]
    )
    return out


# ----------------------------------------------------------------------------
# installments_payments
# ----------------------------------------------------------------------------
def featurize_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """Repayment history: late-payment ratio, days-late, payment consistency."""
    i = inst.copy()
    # DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT > 0  =>  paid late
    i["PAY_DAYS_LATE"] = (i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]).clip(lower=0)
    i["IS_LATE"] = (i["PAY_DAYS_LATE"] > 0).astype(np.int8)
    i["PAY_RATIO"] = _safe_div(i["AMT_PAYMENT"], i["AMT_INSTALMENT"])
    i["UNDERPAID"] = (i["AMT_PAYMENT"] < i["AMT_INSTALMENT"]).astype(np.int8)

    specs = [
        ("INSTALMENT_COUNT", ID_CURR, "count"),
        ("INSTALMENT_LATE_COUNT", "IS_LATE", "sum"),
        ("INSTALMENT_DAYS_LATE_MEAN", "PAY_DAYS_LATE", "mean"),
        ("INSTALMENT_DAYS_LATE_MAX", "PAY_DAYS_LATE", "max"),
        ("INSTALMENT_PAYMENT_RATIO_MEAN", "PAY_RATIO", "mean"),
        ("INSTALMENT_PAYMENT_RATIO_STD", "PAY_RATIO", "std"),
        ("INSTALMENT_UNDERPAID_COUNT", "UNDERPAID", "sum"),
    ]
    out = _agg_by_curr(i, specs)
    out["INSTALMENT_LATE_RATIO"] = _safe_div(out["INSTALMENT_LATE_COUNT"], out["INSTALMENT_COUNT"])
    out["INSTALMENT_UNDERPAID_RATIO"] = _safe_div(
        out["INSTALMENT_UNDERPAID_COUNT"], out["INSTALMENT_COUNT"]
    )
    return out


# ----------------------------------------------------------------------------
# POS_CASH_balance + credit_card_balance
# ----------------------------------------------------------------------------
def featurize_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    """POS/cash loan balances: DPD stats, instalment progress, contract status mix."""
    p = pos.copy()
    p["_active"] = (p["NAME_CONTRACT_STATUS"] == "Active").astype(np.int8)

    specs = [
        ("POS_MONTH_COUNT", ID_CURR, "count"),
        ("POS_DPD_MAX", "SK_DPD", "max"),
        ("POS_DPD_MEAN", "SK_DPD", "mean"),
        ("POS_DPD_DEF_MAX", "SK_DPD_DEF", "max"),
        ("POS_CNT_INSTALMENT_MEAN", "CNT_INSTALMENT", "mean"),
        ("POS_CNT_INSTALMENT_FUTURE_MEAN", "CNT_INSTALMENT_FUTURE", "mean"),
        ("POS_ACTIVE_COUNT", "_active", "sum"),
    ]
    out = _agg_by_curr(p, specs)
    # remaining-instalment ratio (higher => loan still early in life)
    out["POS_FUTURE_INSTALMENT_RATIO"] = _safe_div(
        out["POS_CNT_INSTALMENT_FUTURE_MEAN"], out["POS_CNT_INSTALMENT_MEAN"]
    )
    out["POS_ACTIVE_RATIO"] = _safe_div(out["POS_ACTIVE_COUNT"], out["POS_MONTH_COUNT"])
    return out


def featurize_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    """Credit card balances: utilization, drawings, DPD, payment behavior."""
    specs = [
        ("CC_MONTH_COUNT", ID_CURR, "count"),
        ("CC_AMT_BALANCE_MEAN", "AMT_BALANCE", "mean"),
        ("CC_AMT_BALANCE_MAX", "AMT_BALANCE", "max"),
        ("CC_CREDIT_LIMIT_MEAN", "AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        ("CC_CREDIT_LIMIT_MAX", "AMT_CREDIT_LIMIT_ACTUAL", "max"),
        ("CC_DPD_MAX", "SK_DPD", "max"),
        ("CC_DPD_MEAN", "SK_DPD", "mean"),
        ("CC_DPD_DEF_MAX", "SK_DPD_DEF", "max"),
        ("CC_DRAWINGS_ATM_MEAN", "AMT_DRAWINGS_ATM_CURRENT", "mean"),
        ("CC_AMT_PAYMENT_TOTAL_MEAN", "AMT_PAYMENT_TOTAL_CURRENT", "mean"),
    ]
    out = _agg_by_curr(cc, specs)
    out["CC_UTILIZATION_MEAN"] = _safe_div(out["CC_AMT_BALANCE_MEAN"], out["CC_CREDIT_LIMIT_MEAN"])
    out["CC_UTILIZATION_MAX"] = _safe_div(out["CC_AMT_BALANCE_MAX"], out["CC_CREDIT_LIMIT_MAX"])
    out["CC_PAYMENT_VS_BALANCE"] = _safe_div(
        out["CC_AMT_PAYMENT_TOTAL_MEAN"], out["CC_AMT_BALANCE_MEAN"]
    )
    return out


# ----------------------------------------------------------------------------
# application-level ratios
# ----------------------------------------------------------------------------
def add_application_features(app: pd.DataFrame) -> pd.DataFrame:
    """Ratios on the main table itself. EXT_SOURCE_1/2/3 are retained as-is.

    When the external-source columns are present, also adds interaction
    features (product / sum / square): their combinations carry signal that is
    non-redundant with the individual scores on the real data
    (EXT2xEXT3 |r|=0.176 vs 0.155-0.179 for the singles).
    """
    a = app.copy()
    a["DTI_RATIO"] = _safe_div(a["AMT_CREDIT"], a["AMT_INCOME_TOTAL"])  # debt-to-income
    a["CREDIT_ANNUITY_RATIO"] = _safe_div(a["AMT_CREDIT"], a["AMT_ANNUITY"])
    a["ANNUITY_INCOME_RATIO"] = _safe_div(a["AMT_ANNUITY"], a["AMT_INCOME_TOTAL"])
    a["CREDIT_GOODS_RATIO"] = _safe_div(a["AMT_CREDIT"], a["AMT_GOODS_PRICE"])
    a["CHILDREN_RATIO"] = _safe_div(a["CNT_CHILDREN"], a["CNT_FAM_MEMBERS"])
    a["AGE_YEARS"] = -a["DAYS_BIRTH"] / 365.25
    # 365243 = 'pension/unemployed' sentinel -> NaN rather than 1000 years
    a["EMPLOYMENT_YEARS"] = (-a["DAYS_EMPLOYED"] / 365.25).where(a["DAYS_EMPLOYED"] != 365243)

    # Top Kaggle credit ratio features
    a["PAYMENT_RATE"] = _safe_div(a["AMT_ANNUITY"], a["AMT_CREDIT"])
    a["INCOME_PER_PERSON"] = _safe_div(a["AMT_INCOME_TOTAL"], a["CNT_FAM_MEMBERS"])
    a["EMPLOYED_TO_AGE_RATIO"] = _safe_div(a["EMPLOYMENT_YEARS"], a["AGE_YEARS"])

    if {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}.issubset(a.columns):
        ext1, ext2, ext3 = a["EXT_SOURCE_1"], a["EXT_SOURCE_2"], a["EXT_SOURCE_3"]
        ext_df = a[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
        
        # Interactions & statistical composites
        a["EXT_SOURCES_PROD_2_3"] = ext2 * ext3
        a["EXT_SOURCES_PROD_1_2_3"] = ext1 * ext2 * ext3
        a["EXT_SOURCES_SUM"] = ext1.fillna(0.0) + ext2.fillna(0.0) + ext3.fillna(0.0)
        a["EXT_SOURCES_MEAN"] = ext_df.mean(axis=1)
        a["EXT_SOURCES_STD"] = ext_df.std(axis=1).fillna(0.0)
        a["EXT_SOURCES_MIN"] = ext_df.min(axis=1)
        a["EXT_SOURCES_MAX"] = ext_df.max(axis=1)
        a["EXT_SOURCES_WEIGHTED"] = (ext1.fillna(0.0) * 2.0 + ext2.fillna(0.0) * 3.0 + ext3.fillna(0.0) * 4.0) / 9.0
        a["EXT_SOURCES_2_SQ"] = ext2 ** 2
    return a


def fit_organization_type_encoding(app_train: pd.DataFrame) -> Dict[str, float]:
    """Learn smoothed mean-TARGET per ORGANIZATION_TYPE category (train ONLY).

    encoded(cat) = (mean_cat * n_cat + smoothing * prior) / (n_cat + smoothing)

    Must be fitted on the training table exclusively (it reads TARGET), then
    applied to train and test alike - same protocol as the preprocessors.
    """
    if ORG_TYPE_COL not in app_train.columns:
        return {}
    if TARGET not in app_train.columns:
        raise ValueError("fit_organization_type_encoding requires the TARGET column (train table).")
    prior = float(app_train[TARGET].mean())
    stats = app_train.groupby(ORG_TYPE_COL, observed=True)[TARGET].agg(["mean", "count"])
    s = ORG_TE_SMOOTHING
    mapping = {
        str(cat): float((row["mean"] * row["count"] + s * prior) / (row["count"] + s))
        for cat, row in stats.iterrows()
    }
    mapping["__prior__"] = prior
    logger.info("ORGANIZATION_TYPE target encoding fitted: %d categories, prior=%.4f", len(mapping) - 1, prior)
    return mapping


def add_organization_type_encoding(app: pd.DataFrame, mapping: Dict[str, float]) -> pd.DataFrame:
    """Apply a fitted ORGANIZATION_TYPE encoding; unseen/missing -> prior."""
    if not mapping or ORG_TYPE_COL not in app.columns:
        return app
    a = app.copy()
    prior = float(mapping.get("__prior__", 0.0))
    te = a[ORG_TYPE_COL].astype(object).map(mapping)
    a[ORG_TYPE_ENCODED] = pd.to_numeric(te, errors="coerce").fillna(prior).astype("float32")
    return a


# ----------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------
def build_feature_table(
    tables: Dict[str, pd.DataFrame], verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """Featurize all satellite tables and merge onto the application tables.

    tables: dict as produced by HomeCreditDataLoader.load_all() (must include
    'application_train' and 'application_test'; satellite tables optional).

    Returns
    -------
    (feature_df, y)
        feature_df : one row per SK_ID_CURR (train+test concatenated)
        y          : TARGET series aligned positionally (NaN for test rows)
    """
    train = add_application_features(tables["application_train"])
    te_mapping = fit_organization_type_encoding(train)  # train-only fit (no leakage)
    test = add_application_features(tables["application_test"])
    train = add_organization_type_encoding(train, te_mapping)
    test = add_organization_type_encoding(test, te_mapping)

    y_train = train[TARGET].astype(np.int8)
    y_test = pd.Series(np.nan, index=test.index)
    y = pd.concat([y_train, y_test], axis=0, ignore_index=True)

    train = train.drop(columns=[TARGET])  # application_test has no TARGET by design
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    if verbose:
        logger.info("Application features: %d rows, %d cols", len(combined), combined.shape[1])

    featurizers = {
        "bureau": lambda: featurize_bureau(tables["bureau"], tables.get("bureau_balance")),
        "previous_application": lambda: featurize_previous_application(tables["previous_application"]),
        "installments_payments": lambda: featurize_installments(tables["installments_payments"]),
        "POS_CASH_balance": lambda: featurize_pos_cash(tables["POS_CASH_balance"]),
        "credit_card_balance": lambda: featurize_credit_card(tables["credit_card_balance"]),
    }
    for name, featurize in featurizers.items():
        if name not in tables:
            if verbose:
                logger.warning("Table '%s' not loaded - skipping its features", name)
            continue
        feats = featurize()
        combined = combined.merge(feats, on=ID_CURR, how="left")
        if verbose:
            logger.info("Merged %-22s -> +%d features", name, feats.shape[1] - 1)

    return combined, y


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    from data_loader import HomeCreditDataLoader

    loader = HomeCreditDataLoader(data_dir="data")
    tables = loader.load_all()
    feats, y = build_feature_table(tables)
    print(f"Feature table: {feats.shape}")
    print(f"TARGET: {y.value_counts(dropna=False).to_dict()}")
