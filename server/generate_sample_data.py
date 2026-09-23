"""Generates a realistic synthetic dataset for Home Credit Default Risk.

Configured by default to generate exactly 100,000 total rows across all 8 tables.
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

# Default row counts calibrated to sum to exactly 100,000 (100K) rows
DEFAULT_100K_ROW_COUNTS = {
    "application_train": 10000,
    "application_test": 2000,
    "bureau": 15000,
    "bureau_balance": 25000,
    "previous_application": 10000,
    "POS_CASH_balance": 15000,
    "installments_payments": 15000,
    "credit_card_balance": 8000,
}


def generate_synthetic_home_credit(
    target_dir: str | Path = "data",
    row_counts: Optional[Dict[str, int]] = None,
    seed: int = 42,
):
    counts = row_counts or DEFAULT_100K_ROW_COUNTS
    total_requested = sum(counts.values())

    np.random.seed(seed)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic Home Credit dataset ({total_requested:,} total rows) in: {target_path.resolve()}...")

    n_train = counts["application_train"]
    n_test = counts["application_test"]
    n_bureau = counts["bureau"]
    n_bureau_balance = counts["bureau_balance"]
    n_prev = counts["previous_application"]
    n_pos = counts["POS_CASH_balance"]
    n_inst = counts["installments_payments"]
    n_cc = counts["credit_card_balance"]

    # 1. Primary application IDs
    train_ids = np.arange(100000, 100000 + n_train)
    test_ids = np.arange(200000, 200000 + n_test)
    all_app_ids = np.concatenate([train_ids, test_ids])

    # 1. application_train.csv
    # Imbalanced target (approx 8.07% default rate like real Kaggle dataset)
    target_labels = np.random.choice([0, 1], size=n_train, p=[0.9193, 0.0807])
    train_df = pd.DataFrame(
        {
            "SK_ID_CURR": train_ids,
            "TARGET": target_labels,
            "NAME_CONTRACT_TYPE": np.random.choice(["Cash loans", "Revolving loans"], size=n_train, p=[0.9, 0.1]),
            "CODE_GENDER": np.random.choice(["M", "F"], size=n_train, p=[0.35, 0.65]),
            "FLAG_OWN_CAR": np.random.choice(["Y", "N"], size=n_train),
            "FLAG_OWN_REALTY": np.random.choice(["Y", "N"], size=n_train),
            "CNT_CHILDREN": np.random.poisson(0.5, size=n_train),
            "AMT_INCOME_TOTAL": np.random.exponential(150000, size=n_train).round(2),
            "AMT_CREDIT": np.random.exponential(500000, size=n_train).round(2),
            "AMT_ANNUITY": np.random.exponential(25000, size=n_train).round(2),
            "AMT_GOODS_PRICE": np.random.exponential(450000, size=n_train).round(2),
            "EXT_SOURCE_1": np.random.uniform(0.1, 0.9, size=n_train),
            "EXT_SOURCE_2": np.random.uniform(0.1, 0.9, size=n_train),
            "EXT_SOURCE_3": np.random.uniform(0.1, 0.9, size=n_train),
            "DAYS_BIRTH": -np.random.randint(7000, 25000, size=n_train),
            "DAYS_EMPLOYED": -np.random.randint(100, 10000, size=n_train),
            "OWN_CAR_AGE": np.random.randint(1, 30, size=n_train).astype(float),
        }
    )
    # Inject realistic missing values
    train_df.loc[np.random.choice(n_train, size=int(n_train * 0.65), replace=False), "OWN_CAR_AGE"] = np.nan
    train_df.loc[np.random.choice(n_train, size=int(n_train * 0.55), replace=False), "EXT_SOURCE_1"] = np.nan
    train_df.loc[np.random.choice(n_train, size=int(n_train * 0.20), replace=False), "EXT_SOURCE_3"] = np.nan
    train_df.loc[np.random.choice(n_train, size=int(n_train * 0.05), replace=False), "AMT_ANNUITY"] = np.nan
    train_df.to_csv(target_path / "application_train.csv", index=False)

    # 2. application_test.csv
    test_df = pd.DataFrame(
        {
            "SK_ID_CURR": test_ids,
            "NAME_CONTRACT_TYPE": np.random.choice(["Cash loans", "Revolving loans"], size=n_test, p=[0.9, 0.1]),
            "CODE_GENDER": np.random.choice(["M", "F"], size=n_test, p=[0.35, 0.65]),
            "FLAG_OWN_CAR": np.random.choice(["Y", "N"], size=n_test),
            "FLAG_OWN_REALTY": np.random.choice(["Y", "N"], size=n_test),
            "CNT_CHILDREN": np.random.poisson(0.5, size=n_test),
            "AMT_INCOME_TOTAL": np.random.exponential(150000, size=n_test).round(2),
            "AMT_CREDIT": np.random.exponential(500000, size=n_test).round(2),
            "AMT_ANNUITY": np.random.exponential(25000, size=n_test).round(2),
            "AMT_GOODS_PRICE": np.random.exponential(450000, size=n_test).round(2),
            "EXT_SOURCE_1": np.random.uniform(0.1, 0.9, size=n_test),
            "EXT_SOURCE_2": np.random.uniform(0.1, 0.9, size=n_test),
            "EXT_SOURCE_3": np.random.uniform(0.1, 0.9, size=n_test),
            "DAYS_BIRTH": -np.random.randint(7000, 25000, size=n_test),
            "DAYS_EMPLOYED": -np.random.randint(100, 10000, size=n_test),
            "OWN_CAR_AGE": np.random.randint(1, 30, size=n_test).astype(float),
        }
    )
    test_df.loc[np.random.choice(n_test, size=int(n_test * 0.65), replace=False), "OWN_CAR_AGE"] = np.nan
    test_df.loc[np.random.choice(n_test, size=int(n_test * 0.55), replace=False), "EXT_SOURCE_1"] = np.nan
    test_df.loc[np.random.choice(n_test, size=int(n_test * 0.20), replace=False), "EXT_SOURCE_3"] = np.nan
    test_df.loc[np.random.choice(n_test, size=int(n_test * 0.05), replace=False), "AMT_ANNUITY"] = np.nan
    test_df.to_csv(target_path / "application_test.csv", index=False)

    # 3. bureau.csv
    bureau_curr_ids = np.random.choice(all_app_ids, size=n_bureau)
    bureau_ids = np.arange(5000000, 5000000 + n_bureau)
    bureau_df = pd.DataFrame(
        {
            "SK_ID_CURR": bureau_curr_ids,
            "SK_ID_BUREAU": bureau_ids,
            "CREDIT_ACTIVE": np.random.choice(["Closed", "Active"], size=n_bureau, p=[0.7, 0.3]),
            "DAYS_CREDIT": -np.random.randint(1, 3000, size=n_bureau),
            "CREDIT_DAY_OVERDUE": np.random.choice([0, 10, 30], size=n_bureau, p=[0.95, 0.03, 0.02]),
            "AMT_CREDIT_MAX_OVERDUE": np.random.exponential(5000, size=n_bureau).round(2),
            "AMT_CREDIT_SUM": np.random.exponential(200000, size=n_bureau).round(2),
            "AMT_CREDIT_SUM_DEBT": np.random.exponential(80000, size=n_bureau).round(2),
        }
    )
    bureau_df.to_csv(target_path / "bureau.csv", index=False)

    # 4. bureau_balance.csv
    bureau_bal_ids = np.random.choice(bureau_ids, size=n_bureau_balance)
    bureau_bal_df = pd.DataFrame(
        {
            "SK_ID_BUREAU": bureau_bal_ids,
            "MONTHS_BALANCE": -np.random.randint(0, 96, size=n_bureau_balance),
            "STATUS": np.random.choice(["C", "0", "1", "2", "X"], size=n_bureau_balance, p=[0.4, 0.4, 0.1, 0.05, 0.05]),
        }
    )
    bureau_bal_df.to_csv(target_path / "bureau_balance.csv", index=False)

    # 5. previous_application.csv
    prev_curr_ids = np.random.choice(all_app_ids, size=n_prev)
    prev_ids = np.arange(1000000, 1000000 + n_prev)
    prev_df = pd.DataFrame(
        {
            "SK_ID_PREV": prev_ids,
            "SK_ID_CURR": prev_curr_ids,
            "NAME_CONTRACT_TYPE": np.random.choice(["Consumer loans", "Cash loans", "Revolving loans"], size=n_prev),
            "AMT_APPLICATION": np.random.exponential(100000, size=n_prev).round(2),
            "AMT_CREDIT": np.random.exponential(120000, size=n_prev).round(2),
            "NAME_CONTRACT_STATUS": np.random.choice(["Approved", "Canceled", "Refused"], size=n_prev, p=[0.6, 0.2, 0.2]),
        }
    )
    prev_df.to_csv(target_path / "previous_application.csv", index=False)

    # 6. POS_CASH_balance.csv
    pos_sub_idx = np.random.choice(n_prev, size=n_pos)
    pos_df = pd.DataFrame(
        {
            "SK_ID_PREV": prev_ids[pos_sub_idx],
            "SK_ID_CURR": prev_curr_ids[pos_sub_idx],
            "MONTHS_BALANCE": -np.random.randint(1, 48, size=n_pos),
            "CNT_INSTALMENT": np.random.choice([6, 12, 24, 36], size=n_pos),
            "CNT_INSTALMENT_FUTURE": np.random.randint(0, 36, size=n_pos),
            "NAME_CONTRACT_STATUS": "Active",
            "SK_DPD": np.random.choice([0, 5], size=n_pos, p=[0.98, 0.02]),
        }
    )
    pos_df.to_csv(target_path / "POS_CASH_balance.csv", index=False)

    # 7. installments_payments.csv
    inst_sub_idx = np.random.choice(n_prev, size=n_inst)
    inst_df = pd.DataFrame(
        {
            "SK_ID_PREV": prev_ids[inst_sub_idx],
            "SK_ID_CURR": prev_curr_ids[inst_sub_idx],
            "NUM_INSTALMENT_VERSION": 1,
            "NUM_INSTALMENT_NUMBER": np.random.randint(1, 24, size=n_inst),
            "DAYS_INSTALMENT": -np.random.randint(10, 1000, size=n_inst),
            "DAYS_ENTRY_PAYMENT": -np.random.randint(10, 1000, size=n_inst),
            "AMT_INSTALMENT": np.random.exponential(10000, size=n_inst).round(2),
            "AMT_PAYMENT": np.random.exponential(10000, size=n_inst).round(2),
        }
    )
    inst_df.to_csv(target_path / "installments_payments.csv", index=False)

    # 8. credit_card_balance.csv
    cc_sub_idx = np.random.choice(n_prev, size=n_cc)
    cc_df = pd.DataFrame(
        {
            "SK_ID_PREV": prev_ids[cc_sub_idx],
            "SK_ID_CURR": prev_curr_ids[cc_sub_idx],
            "MONTHS_BALANCE": -np.random.randint(1, 36, size=n_cc),
            "AMT_BALANCE": np.random.exponential(50000, size=n_cc).round(2),
            "AMT_CREDIT_LIMIT_ACTUAL": np.random.choice([50000, 100000, 200000], size=n_cc),
            "AMT_DRAWINGS_ATM_CURRENT": np.random.exponential(10000, size=n_cc).round(2),
            "AMT_PAYMENT_TOTAL_CURRENT": np.random.exponential(15000, size=n_cc).round(2),
        }
    )
    cc_df.to_csv(target_path / "credit_card_balance.csv", index=False)

    generated_total = sum(len(df) for df in [train_df, test_df, bureau_df, bureau_bal_df, prev_df, pos_df, inst_df, cc_df])
    print(f"[OK] Generated {generated_total:,} total rows across all 8 tables.")


if __name__ == "__main__":
    generate_synthetic_home_credit()
