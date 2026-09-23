"""Command-line runner for the Home Credit Data Loading & Quality Validation module.

Usage:
    python server/run_loader.py [--data-dir data] [--output-report data/data_quality_report.md] [--nrows 5000]
"""

import argparse
import sys
from pathlib import Path

# Add server directory to path if executed directly
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from data_loader import DATA_FILES, HomeCreditDataLoader


def main():
    parser = argparse.ArgumentParser(
        description="Load Home Credit Default Risk CSVs, downcast dtypes, validate join keys, and generate quality report."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to directory containing Home Credit CSV files (default: 'data')",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="data/data_quality_report.md",
        help="Path to save the generated Markdown report (default: 'data/data_quality_report.md')",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit for rapid inspection/testing (default: None, loads entire file)",
    )
    parser.add_argument(
        "--no-downcast",
        action="store_true",
        help="Disable automatic numeric dtype downcasting",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        help=f"Specific tables to load. Available: {list(DATA_FILES.keys())}",
    )

    args = parser.parse_args()

    data_dir_path = Path(args.data_dir)
    print("=" * 80)
    print("  HOME CREDIT DEFAULT RISK - DATA LOADER & QUALITY VALIDATOR")
    print("=" * 80)
    print(f"Data Directory : {data_dir_path.resolve()}")
    print(f"Output Report  : {Path(args.output_report).resolve()}")
    print(f"Memory Downcast: {'Disabled' if args.no_downcast else 'Enabled'}")
    print(f"Row Limit      : {args.nrows or 'Full Dataset'}")
    print("=" * 80)

    loader = HomeCreditDataLoader(data_dir=data_dir_path)

    # Check available files in directory
    existing_tables = []
    missing_tables = []
    for tname, fname in DATA_FILES.items():
        if (data_dir_path / fname).is_file():
            existing_tables.append(tname)
        else:
            missing_tables.append(fname)

    if not existing_tables:
        print(f"\n[!] No Home Credit CSV files found in '{data_dir_path.resolve()}'.")
        print("\nExpected CSV files:")
        for fname in DATA_FILES.values():
            print(f"  - {fname}")
        print("\nPlease download the dataset from Kaggle and place the CSV files inside the 'data/' folder.")
        print("Tip: You can run 'python server/generate_sample_data.py' to generate a small synthetic dataset for testing.")
        sys.exit(1)

    if missing_tables:
        print(f"\n[Note] Found {len(existing_tables)} tables. Missing {len(missing_tables)} tables:")
        for m in missing_tables:
            print(f"  - {m}")
        print("Proceeding with available tables...\n")

    tables_to_load = [t for t in (args.tables or list(DATA_FILES.keys())) if t in existing_tables]

    # Load tables
    loader.load_all(
        tables=tables_to_load,
        downcast=not args.no_downcast,
        nrows=args.nrows,
        ignore_missing=True,
    )

    # Generate and save report
    report = loader.generate_data_quality_report(output_path=args.output_report)

    print("\n" + "=" * 80)
    print("  EXECUTIVE SUMMARY OF DATA QUALITY REPORT")
    print("=" * 80)
    print(report)
    print("\n[OK] Data loading and validation complete!")


if __name__ == "__main__":
    main()
