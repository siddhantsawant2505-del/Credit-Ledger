"""Command-line runner: build features and fit the preprocessing pipeline.

Usage:
    python server/run_preprocessing.py [--data-dir data] [--outdir models]

Steps:
1. Load all tables (full dataset) via HomeCreditDataLoader.
2. Build the merged feature table (train+test featurized together).
3. Fit the ColumnTransformer on TRAIN rows ONLY (no leakage into test).
4. Save:
   - models/preprocessor_scaled.joblib     (for logistic / DNN)
   - models/preprocessor_unscaled.joblib   (for tree models)
   - data/feature_table.parquet            (merged features + TARGET, split col)
   - data/preprocessing_summary.txt        (feature counts + top features)
5. Print class-imbalance helpers (scale_pos_weight, class_weight dict).

The saved feature table keeps train/test rows together with a 'SPLIT' column
so model notebooks can select rows without re-running aggregation.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from data_loader import HomeCreditDataLoader
from feature_engineering import ID_CURR, TARGET, build_feature_table
from preprocessing import (
    build_preprocessor,
    get_class_weights,
    get_scale_pos_weight,
    get_transformed_feature_names,
    summarize_features,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature engineering + preprocessing pipeline for Home Credit Default Risk."
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--outdir", type=str, default="models")
    parser.add_argument("--low-cardinality-max", type=int, default=15)
    parser.add_argument(
        "--skip-transformed",
        action="store_true",
        help="Skip transforming the full matrix (faster smoke runs); summary uses raw correlations.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # ------------------------------------------------------------------ load
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("STEP 1/5: Loading all tables (full dataset)")
    loader = HomeCreditDataLoader(data_dir=data_dir)
    tables = loader.load_all()
    authenticity = loader.check_data_authenticity()
    logger.info(authenticity["message"])
    if not authenticity["is_authentic"]:
        logger.error("Aborting: data is not the real Kaggle dataset.")
        sys.exit(2)

    # -------------------------------------------------------- feature build
    logger.info("=" * 70)
    logger.info("STEP 2/5: Building merged feature table")
    feature_df, y = build_feature_table(tables)
    train_mask = y.notna().values
    logger.info(
        "Feature table: %d rows x %d cols | train rows: %d | test rows: %d",
        len(feature_df),
        feature_df.shape[1],
        int(train_mask.sum()),
        int((~train_mask).sum()),
    )

    # ------------------------------------------------- fit preprocessors
    logger.info("=" * 70)
    logger.info("STEP 3/5: Fitting preprocessors on TRAIN rows only")
    df_train = feature_df.loc[train_mask].reset_index(drop=True)

    pre_unscaled = build_preprocessor(
        df_train, scale_numeric=False, low_cardinality_max=args.low_cardinality_max
    )
    pre_scaled = build_preprocessor(
        df_train, scale_numeric=True, low_cardinality_max=args.low_cardinality_max
    )
    pre_unscaled.fit(df_train)
    pre_scaled.fit(df_train)

    # ------------------------------------------------------ save artifacts
    logger.info("=" * 70)
    logger.info("STEP 4/5: Saving artifacts")
    import joblib

    joblib.dump(pre_unscaled, outdir / "preprocessor_unscaled.joblib")
    joblib.dump(pre_scaled, outdir / "preprocessor_scaled.joblib")
    logger.info("Saved %s", (outdir / "preprocessor_unscaled.joblib").resolve())
    logger.info("Saved %s", (outdir / "preprocessor_scaled.joblib").resolve())

    feature_out = feature_df.copy()
    feature_out["SPLIT"] = np.where(train_mask, "train", "test")
    feature_out[TARGET] = y.values
    parquet_path = data_dir / "feature_table.parquet"
    feature_out.to_parquet(parquet_path, index=False)
    logger.info("Saved %s (%.1f MB)", parquet_path.resolve(), parquet_path.stat().st_size / 1024 / 1024)

    # ------------------------------------------------------------ summary
    logger.info("=" * 70)
    logger.info("STEP 5/5: Building summary")
    spw = get_scale_pos_weight(y[train_mask].reset_index(drop=True))
    cw = get_class_weights(y[train_mask].reset_index(drop=True))

    summary_parts = [
        "HOME CREDIT DEFAULT RISK - PREPROCESSING SUMMARY",
        "=" * 60,
        f"Raw feature table        : {feature_df.shape[0]:,} rows x {feature_df.shape[1] - 1} features",
        f"Train / test rows        : {int(train_mask.sum()):,} / {int((~train_mask).sum()):,}",
        f"Default rate (train)     : {y[train_mask].mean() * 100:.2f}%",
        f"scale_pos_weight (XGB/LGBM): {spw:.3f}",
        f"class_weight (LR/RF)     : {cw}",
        "",
    ]

    if args.skip_transformed:
        summary_parts.append(summarize_features(feature_df, y, top_n=10))
    else:
        logger.info("Transforming full train matrix with the scaled preprocessor...")
        X_transformed = pre_scaled.transform(df_train)
        names = get_transformed_feature_names(pre_scaled)
        summary_parts.append(
            summarize_features(
                feature_df, y, transformed=X_transformed, transformed_names=names, top_n=10
            )
        )
        # one-hot expansion count for the record
        summary_parts.insert(
            4, f"Transformed feature count: {len(names)}"
        )

    summary_text = "\n".join(summary_parts)
    summary_path = data_dir / "preprocessing_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    logger.info("Saved %s", summary_path.resolve())

    print("\n" + "=" * 70)
    print(summary_text)
    print("=" * 70)
    print(f"\n[OK] Pipeline complete in {time.time() - t0:.0f}s. Ready for model notebooks.")


if __name__ == "__main__":
    main()
