"""Unit tests for feature_engineering.py and preprocessing.py.

Uses small synthetic frames with the exact column schemas of the real tables.
"""

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

server_dir = Path(__file__).resolve().parent
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))

from feature_engineering import (
    add_application_features,
    build_feature_table,
    featurize_bureau,
    featurize_credit_card,
    featurize_installments,
    featurize_previous_application,
    featurize_pos_cash,
)
from preprocessing import (
    build_preprocessor,
    get_class_weights,
    get_scale_pos_weight,
    split_column_types,
)


def make_bureau():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 3, 3],
            "SK_ID_BUREAU": [10, 11, 12, 13, 14],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Closed", "Closed"],
            "DAYS_CREDIT": [-100, -500, -200, -300, -800],
            "CREDIT_DAY_OVERDUE": [0, 30, 0, 0, 60],
            "AMT_CREDIT_MAX_OVERDUE": [0.0, 5000.0, np.nan, 0.0, 1000.0],
            "AMT_CREDIT_SUM": [10000.0, 20000.0, 5000.0, 0.0, 10000.0],
            "AMT_CREDIT_SUM_DEBT": [5000.0, 0.0, 1000.0, 0.0, 2000.0],
        }
    )


def make_bureau_balance():
    return pd.DataFrame(
        {
            "SK_ID_BUREAU": [10, 10, 11, 99],  # 99 is an orphan bureau id
            "MONTHS_BALANCE": [-1, -5, -2, -3],
            "STATUS": ["C", "0", "1", "C"],
        }
    )


def make_prev():
    return pd.DataFrame(
        {
            "SK_ID_PREV": [100, 101, 102, 103],
            "SK_ID_CURR": [1, 1, 2, 3],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved", "Canceled"],
            "AMT_APPLICATION": [100000.0, 50000.0, 200000.0, 80000.0],
            "AMT_CREDIT": [90000.0, 0.0, 180000.0, 0.0],
        }
    )


def make_inst():
    return pd.DataFrame(
        {
            "SK_ID_PREV": [100, 100, 101, 102],
            "SK_ID_CURR": [1, 1, 1, 2],
            "NUM_INSTALMENT_VERSION": [1, 1, 1, 1],
            "NUM_INSTALMENT_NUMBER": [1, 2, 1, 1],
            "DAYS_INSTALMENT": [-100, -70, -40, -90],
            "DAYS_ENTRY_PAYMENT": [-98, -60, -40, -95],
            "AMT_INSTALMENT": [1000.0, 1000.0, 500.0, 2000.0],
            "AMT_PAYMENT": [1000.0, 900.0, 500.0, 2000.0],
        }
    )


def make_pos():
    return pd.DataFrame(
        {
            "SK_ID_PREV": [100, 100, 102, 103],
            "SK_ID_CURR": [1, 1, 2, 3],
            "MONTHS_BALANCE": [-1, -2, -1, -1],
            "CNT_INSTALMENT": [12, 12, 24, 6],
            "CNT_INSTALMENT_FUTURE": [6, 6, 0, 3],
            "NAME_CONTRACT_STATUS": ["Active", "Active", "Completed", "Active"],
            "SK_DPD": [0, 10, 0, 5],
            "SK_DPD_DEF": [0, 0, 0, 0],
        }
    )


def make_cc():
    return pd.DataFrame(
        {
            "SK_ID_PREV": [200, 201, 202],
            "SK_ID_CURR": [1, 1, 2],
            "MONTHS_BALANCE": [-1, -2, -1],
            "AMT_BALANCE": [5000.0, 3000.0, 0.0],
            "AMT_CREDIT_LIMIT_ACTUAL": [10000.0, 10000.0, 20000.0],
            "AMT_DRAWINGS_ATM_CURRENT": [100.0, 0.0, 0.0],
            "AMT_PAYMENT_TOTAL_CURRENT": [5000.0, 0.0, 0.0],
            "SK_DPD": [0, 5, 0],
            "SK_DPD_DEF": [0, 0, 0],
        }
    )


def make_app(name="application_train"):
    # train IDs 1-3; test IDs 4-6 (disjoint, like the real dataset)
    ids = [1, 2, 3] if name == "application_train" else [4, 5, 6]
    data = {
        "SK_ID_CURR": ids,
        "AMT_INCOME_TOTAL": [100000.0, 200000.0, 50000.0],
        "AMT_CREDIT": [200000.0, 100000.0, 150000.0],
        "AMT_ANNUITY": [10000.0, 5000.0, 8000.0],
        "AMT_GOODS_PRICE": [180000.0, 95000.0, 140000.0],
        "CNT_CHILDREN": [1, 0, 2],
        "CNT_FAM_MEMBERS": [3.0, 2.0, 4.0],
        "DAYS_BIRTH": [-10000.0, -15000.0, -8000.0],
        "DAYS_EMPLOYED": [-2000.0, 365243.0, -1000.0],  # middle one = pension sentinel
        "EXT_SOURCE_1": [0.4, np.nan, 0.7],
        "EXT_SOURCE_2": [0.6, 0.5, np.nan],
        "EXT_SOURCE_3": [0.8, 0.3, 0.2],
    }
    if name == "application_train":
        data["TARGET"] = [0, 1, 0]
    return pd.DataFrame(data)


class TestFeatureEngineering(unittest.TestCase):
    def test_safe_div_semantics(self):
        from feature_engineering import _safe_div

        out = _safe_div(pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 0.0, np.nan]))
        self.assertEqual(out.iloc[0], 0.5)
        self.assertTrue(np.isnan(out.iloc[1]))  # zero denominator -> NaN, not inf
        self.assertTrue(np.isnan(out.iloc[2]))

    def test_bureau_counts_and_ratios(self):
        out = featurize_bureau(make_bureau(), make_bureau_balance())
        row1 = out[out["SK_ID_CURR"] == 1].iloc[0]
        self.assertEqual(row1["BUREAU_LOAN_COUNT"], 2)
        self.assertEqual(row1["BUREAU_ACTIVE_COUNT"], 1)
        self.assertEqual(row1["BUREAU_CLOSED_COUNT"], 1)
        self.assertAlmostEqual(row1["BUREAU_ACTIVE_RATIO"], 0.5)
        self.assertEqual(row1["BUREAU_CREDIT_DAY_OVERDUE_MAX"], 30)
        self.assertAlmostEqual(row1["BUREAU_AMT_CREDIT_SUM_TOTAL"], 30000.0)
        # bureau_balance status sums aggregated through SK_ID_BUREAU
        self.assertEqual(row1["BUREAU_BB_STATUS_C_SUM"], 1)  # bureau 10's two rows
        self.assertEqual(row1["BUREAU_BB_STATUS_1_SUM"], 1)  # bureau 11
        # orphan bureau id 99 must not appear anywhere
        self.assertNotIn("BUREAU_BB_STATUS_C_SUM", [])
        self.assertEqual(int(out["SK_ID_CURR"].nunique()), 3)

    def test_bureau_without_balance(self):
        out = featurize_bureau(make_bureau(), None)
        self.assertNotIn("BUREAU_BAL_MONTHS_MIN", out.columns)
        self.assertIn("BUREAU_LOAN_COUNT", out.columns)

    def test_prev_approval_rate(self):
        out = featurize_previous_application(make_prev())
        row1 = out[out["SK_ID_CURR"] == 1].iloc[0]
        self.assertEqual(row1["PREV_APP_COUNT"], 2)
        self.assertAlmostEqual(row1["PREV_APPROVAL_RATE"], 0.5)
        self.assertAlmostEqual(row1["PREV_REFUSAL_RATE"], 0.5)
        # granted/requested = mean(90k, 0) / mean(100k, 50k)
        self.assertAlmostEqual(row1["PREV_GRANTED_VS_REQUESTED"], 45000.0 / 75000.0)

    def test_installments_late(self):
        out = featurize_installments(make_inst())
        row1 = out[out["SK_ID_CURR"] == 1].iloc[0]
        self.assertEqual(row1["INSTALMENT_COUNT"], 3)
        # late rows: (DAYS_ENTRY - DAYS_INSTAL) = 2 and 10 -> late; -0 not late
        self.assertEqual(row1["INSTALMENT_LATE_COUNT"], 2)
        self.assertAlmostEqual(row1["INSTALMENT_LATE_RATIO"], 2 / 3)
        self.assertAlmostEqual(row1["INSTALMENT_DAYS_LATE_MAX"], 10.0)
        # underpaid: 900 < 1000 -> 1 underpaid
        self.assertEqual(row1["INSTALMENT_UNDERPAID_COUNT"], 1)

    def test_pos_and_cc(self):
        pos = featurize_pos_cash(make_pos())
        row1 = pos[pos["SK_ID_CURR"] == 1].iloc[0]
        self.assertEqual(row1["POS_DPD_MAX"], 10)
        self.assertAlmostEqual(row1["POS_ACTIVE_RATIO"], 1.0)

        cc = featurize_credit_card(make_cc())
        row1 = cc[cc["SK_ID_CURR"] == 1].iloc[0]
        self.assertAlmostEqual(row1["CC_UTILIZATION_MEAN"], 4000.0 / 10000.0)
        self.assertAlmostEqual(row1["CC_UTILIZATION_MAX"], 5000.0 / 10000.0)

    def test_application_features_sentinel(self):
        out = add_application_features(make_app())
        row2 = out[out["SK_ID_CURR"] == 2].iloc[0]
        self.assertAlmostEqual(row2["DTI_RATIO"], 0.5)
        self.assertTrue(np.isnan(row2["EMPLOYMENT_YEARS"]))  # 365243 sentinel -> NaN
        row1 = out[out["SK_ID_CURR"] == 1].iloc[0]
        self.assertAlmostEqual(row1["EMPLOYMENT_YEARS"], 2000 / 365.25, places=3)

    def test_application_features_ext_interactions(self):
        out = add_application_features(make_app())
        row1 = out[out["SK_ID_CURR"] == 1].iloc[0]
        row2 = out[out["SK_ID_CURR"] == 2].iloc[0]
        row3 = out[out["SK_ID_CURR"] == 3].iloc[0]
        # product / square stay NaN when an operand is missing
        self.assertAlmostEqual(row1["EXT_SOURCES_PROD_2_3"], 0.6 * 0.8)
        self.assertAlmostEqual(row2["EXT_SOURCES_PROD_2_3"], 0.5 * 0.3)
        self.assertTrue(np.isnan(row3["EXT_SOURCES_PROD_2_3"]))
        self.assertAlmostEqual(row1["EXT_SOURCES_2_SQ"], 0.36)
        self.assertTrue(np.isnan(row3["EXT_SOURCES_2_SQ"]))
        # sum treats missing scores as 0 (all-no-signal end of the [0,1] range)
        self.assertAlmostEqual(row1["EXT_SOURCES_SUM"], 1.8)
        self.assertAlmostEqual(row2["EXT_SOURCES_SUM"], 0.8)  # missing EXT_1 -> 0
        self.assertAlmostEqual(row3["EXT_SOURCES_SUM"], 0.9)  # missing EXT_2 -> 0

    def test_organization_type_encoding_no_leakage_and_smoothing(self):
        from feature_engineering import (
            fit_organization_type_encoding,
            add_organization_type_encoding,
            ORG_TYPE_ENCODED,
        )

        rng = np.random.default_rng(7)
        n = 5000
        app = pd.DataFrame(
            {
                "SK_ID_CURR": np.arange(n),
                "ORGANIZATION_TYPE": np.where(rng.random(n) < 0.5, "A", "B"),
                "TARGET": np.where(rng.random(n) < 0.3, 1, 0).astype(np.int8),
            }
        )
        app.loc[app["ORGANIZATION_TYPE"] == "A", "TARGET"] = np.where(
            rng.random((app["ORGANIZATION_TYPE"] == "A").sum()) < 0.5, 1, 0
        ).astype(np.int8)  # A defaults ~50%, B ~ default rate stays low

        mapping = fit_organization_type_encoding(app)
        prior = mapping["__prior__"]
        self.assertGreater(mapping["A"], mapping["B"])  # risky category encodes higher
        # smoothing: small-sample category means shrink toward the prior
        self.assertLess(mapping["A"], app.loc[app.ORGANIZATION_TYPE == "A", "TARGET"].mean())

        train_out = add_organization_type_encoding(app, mapping)
        test_app = app.drop(columns=["TARGET"])  # test rows have no TARGET
        test_out = add_organization_type_encoding(test_app, mapping)
        for out in (train_out, test_out):
            self.assertIn(ORG_TYPE_ENCODED, out.columns)
            self.assertTrue(out[ORG_TYPE_ENCODED].between(0, 1).all())
        # unseen category falls back to the prior
        unseen = add_organization_type_encoding(
            pd.DataFrame({"ORGANIZATION_TYPE": ["ZZZ"]}), mapping
        )
        self.assertAlmostEqual(float(unseen[ORG_TYPE_ENCODED].iloc[0]), prior, places=6)

    def test_build_feature_table_skips_encoding_without_org_column(self):
        tables = {
            "application_train": make_app("application_train"),
            "application_test": make_app("application_test"),
        }
        feats, y = build_feature_table(tables, verbose=False)
        self.assertNotIn("ORGANIZATION_TYPE_TE", feats.columns)

    def test_application_features_without_ext_columns(self):
        # backward compatibility: tables lacking the EXT columns must not crash
        out = add_application_features(make_app().drop(columns=["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]))
        self.assertNotIn("EXT_SOURCES_PROD_2_3", out.columns)
        self.assertNotIn("EXT_SOURCES_SUM", out.columns)
        self.assertNotIn("EXT_SOURCES_2_SQ", out.columns)
        self.assertIn("DTI_RATIO", out.columns)

    def test_build_feature_table_end_to_end(self):
        tables = {
            "application_train": make_app("application_train"),
            "application_test": make_app("application_test"),
            "bureau": make_bureau(),
            "bureau_balance": make_bureau_balance(),
            "previous_application": make_prev(),
            "installments_payments": make_inst(),
            "POS_CASH_balance": make_pos(),
            "credit_card_balance": make_cc(),
        }
        feats, y = build_feature_table(tables, verbose=False)

        # 3 train + 3 test rows, one per SK_ID_CURR
        self.assertEqual(len(feats), 6)
        self.assertEqual(feats["SK_ID_CURR"].nunique(), 6)
        self.assertEqual(len(y), 6)
        self.assertEqual(y.notna().sum(), 3)
        self.assertTrue(y.iloc[:3].isin([0, 1]).all())
        self.assertTrue(y.iloc[3:].isna().all())
        # left join keeps applicant 3 even though it has no credit-card rows
        row3 = feats[feats["SK_ID_CURR"] == 3].iloc[0]
        self.assertTrue(pd.isna(row3["CC_MONTH_COUNT"]))
        # TARGET column itself must not leak into features
        self.assertNotIn("TARGET", feats.columns)


class TestPreprocessor(unittest.TestCase):
    def _sample_frame(self):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "SK_ID_CURR": np.arange(100000, 100000 + n),
                "num_a": rng.normal(0, 1, n),
                "num_b": rng.normal(10, 5, n),
                "num_with_nan": np.where(rng.random(n) < 0.2, np.nan, rng.normal(0, 1, n)),
                "cat_low": rng.choice(["A", "B", "C"], n),
                "cat_high": rng.choice([f"org_{i}" for i in range(40)], n),
            }
        )
        # give num_with_nan a train median different from test median (leakage check)
        df.loc[0:99, "num_with_nan"] = 1.0  # train half
        df.loc[100:, "num_with_nan"] = 50.0  # test half
        return df

    def test_split_column_types(self):
        df = self._sample_frame()
        num, cat = split_column_types(df)
        self.assertIn("num_a", num)
        self.assertIn("cat_low", cat)
        self.assertNotIn("SK_ID_CURR", num + cat)

    def test_fit_transform_shapes_and_no_nan(self):
        df = self._sample_frame()
        train = df.iloc[:100]
        pre = build_preprocessor(train, scale_numeric=True)
        Xtr = pre.fit_transform(train)
        Xte = pre.transform(df.iloc[100:])

        self.assertEqual(Xtr.shape[0], 100)
        self.assertEqual(Xtr.shape[1], Xte.shape[1])
        self.assertFalse(np.isnan(Xtr).any())
        self.assertFalse(np.isnan(Xte).any())

    def test_median_fit_on_train_only(self):
        """Test-row NaNs must be imputed with the TRAIN median, not the test median."""
        df = self._sample_frame()
        train = df.iloc[:100]
        test = df.iloc[100:]
        pre = build_preprocessor(train, scale_numeric=False)
        pre.fit(train)
        names = list(pre.get_feature_names_out())
        Xte = pre.transform(test)
        col = names.index("num__num_with_nan")
        # train median of num_with_nan is 1.0; a NaN in test must map to 1.0
        test_with_nan = test.copy()
        test_with_nan["num_with_nan"] = np.nan
        Xte_nan = pre.transform(test_with_nan)
        self.assertAlmostEqual(Xte_nan[0, col], 1.0, places=6)

    def test_scaled_vs_unscaled_differ(self):
        df = self._sample_frame()
        train = df.iloc[:100]
        X_unscaled = build_preprocessor(train, scale_numeric=False).fit_transform(train)
        X_scaled = build_preprocessor(train, scale_numeric=True).fit_transform(train)
        self.assertFalse(np.allclose(X_unscaled, X_scaled))
        # scaled numeric block should have ~0 mean
        names_unscaled = list(
            build_preprocessor(train, scale_numeric=False).fit(train).get_feature_names_out()
        )
        col = names_unscaled.index("num__num_a")
        self.assertAlmostEqual(X_scaled[:, col].mean(), 0.0, places=6)

    def test_cardinality_split(self):
        df = self._sample_frame()
        train = df.iloc[:100]
        pre = build_preprocessor(train, low_cardinality_max=15)
        pre.fit(train)
        names = list(pre.get_feature_names_out())
        # cat_low (3 values) -> one-hot columns
        self.assertTrue(any(n.startswith("cat_low__cat_low_A") for n in names))
        # cat_high (40 values) -> ordinal single column
        self.assertTrue(any(n == "cat_high__cat_high" for n in names))

    def test_imbalance_helpers(self):
        y = pd.Series([0] * 92 + [1] * 8)  # n = 100, k = 2
        spw = get_scale_pos_weight(y)
        self.assertAlmostEqual(spw, 92 / 8)
        cw = get_class_weights(y)
        self.assertAlmostEqual(cw[1], 100 / (2 * 8))
        self.assertAlmostEqual(cw[0], 100 / (2 * 92))


if __name__ == "__main__":
    unittest.main()
