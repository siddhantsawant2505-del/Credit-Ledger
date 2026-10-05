"""Fetch the real Home Credit Default Risk dataset from Kaggle into data/.

Prerequisites (one-time):
1. Accept the competition rules at https://www.kaggle.com/c/home-credit-default-risk
   (click "Join Competition" / "I Understand and Accept").
2. Create an API token at https://www.kaggle.com/settings -> "Create New Token".
   This downloads kaggle.json.
3. Place kaggle.json at ~/.kaggle/kaggle.json (this script checks common Windows/
   Unix locations and can fix permissions if needed).

Usage:
    python server/fetch_data.py [--dest data] [--force]

The script downloads home-credit-default-risk.zip (~72 MB), extracts the 8 CSVs
into the destination directory, verifies all expected files exist, and runs the
loader's authenticity check so you know the data is real before preprocessing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from data_loader import DATA_FILES, HomeCreditDataLoader  # noqa: E402

COMPETITION_SLUG = "home-credit-default-risk"


def find_kaggle_token() -> Path | None:
    """Locate kaggle.json in the standard locations (handles Windows + Unix)."""
    candidates = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json"
        if os.environ.get("KAGGLE_CONFIG_DIR")
        else None,
        Path.cwd() / "kaggle.json",
        current_dir / "kaggle.json",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    return None


def check_token_permissions(token_path: Path) -> None:
    """Warn if kaggle.json is group/world readable (Kaggle API rejects these)."""
    try:
        mode = token_path.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            try:
                token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
                print(f"[OK] Tightened permissions on {token_path} to 600.")
            except OSError:
                print(
                    f"[WARN] {token_path} is readable by other users; the Kaggle "
                    "API may refuse it. Consider chmod 600."
                )
    except OSError:
        pass


def ensure_kaggle_installed() -> bool:
    """Check for the kaggle package; offer to pip-install it into the current env."""
    try:
        import kaggle  # noqa: F401

        return True
    except ImportError:
        pass
    except Exception as err:  # kaggle raises on missing token at import time
        print(f"[WARN] kaggle package present but failed to import: {err}")

    answer = input(
        "The 'kaggle' package is not installed. Install it now with pip? [Y/n] "
    ).strip().lower()
    if answer not in ("", "y", "yes"):
        print("Aborting. Install manually with: pip install kaggle")
        return False

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "kaggle"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] pip install failed:\n{result.stderr}")
        return False
    print("[OK] kaggle package installed.")
    return True


def download_and_extract(dest_dir: Path, force: bool) -> list[Path]:
    """Download the competition zip and extract the 8 expected CSVs into dest_dir."""
    kagglehub_available = False
    try:
        import kagglehub  # type: ignore

        kagglehub_available = True
    except ImportError:
        kagglehub_available = False

    existing = [dest_dir / f for f in DATA_FILES.values() if (dest_dir / f).is_file()]
    if len(existing) == len(DATA_FILES) and not force:
        print(f"[OK] All {len(DATA_FILES)} CSVs already present in {dest_dir}.")
        return existing
    if existing and not force:
        missing = sorted(set(DATA_FILES.values()) - {p.name for p in existing})
        print(f"[INFO] {len(existing)}/{len(DATA_FILES)} CSVs present; missing: {missing}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hcdr_download_") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "home-credit-default-risk.zip"

        print(f"Downloading '{COMPETITION_SLUG}' from Kaggle (~72 MB)...")
        try:
            if kagglehub_available:
                # kagglehub cannot fetch competition files directly in all versions;
                # fall back to the legacy CLI-compatible client for reliability.
                raise RuntimeError("competition files require the kaggle client")
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

            api = KaggleApi()
            api.authenticate()
            api.competition_download_files(COMPETITION_SLUG, path=str(tmp_path), quiet=True)
        except Exception as client_err:
            # Fallback: invoke the kaggle CLI as a subprocess (respects kaggle.json).
            cli = shutil.which("kaggle")
            cmd = [
                *([cli] if cli else [sys.executable, "-m", "kaggle"]),
                "competitions",
                "download",
                "-c",
                COMPETITION_SLUG,
                "-p",
                str(tmp_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                stderr = (result.stderr or result.stdout or "").strip()
                if "403" in stderr or "Forbidden" in stderr or "not accepted" in stderr.lower():
                    print(
                        "[ERROR] Kaggle returned 403 - you must accept the competition "
                        f"rules first: https://www.kaggle.com/c/{COMPETITION_SLUG}/rules"
                    )
                else:
                    print(f"[ERROR] Kaggle download failed:\n{stderr}")
                sys.exit(1)

        if not zip_path.exists():
            # The CLI may have written the zip into tmp under its own name.
            zips = list(tmp_path.glob("*.zip"))
            if not zips:
                print("[ERROR] Download finished but no zip file was found.")
                sys.exit(1)
            zip_path = zips[0]

        print(f"Extracting {zip_path.name} -> {dest_dir.resolve()}...")
        with zipfile.ZipFile(zip_path) as zf:
            expected = set(DATA_FILES.values())
            for member in zf.namelist():
                name = Path(member).name
                if name in expected:
                    zf.extract(member, dest_dir)

    extracted = [dest_dir / f for f in DATA_FILES.values() if (dest_dir / f).is_file()]
    missing = sorted(set(DATA_FILES.values()) - {p.name for p in extracted})
    if missing:
        print(f"[ERROR] Extraction incomplete. Missing files: {missing}")
        sys.exit(1)
    print(f"[OK] All {len(extracted)} CSVs extracted to {dest_dir.resolve()}.")
    return extracted


def validate_against_loader(dest_dir: Path) -> bool:
    """Load the freshly downloaded CSVs and run the authenticity guard."""
    print("\nValidating downloaded data with the loader...")
    loader = HomeCreditDataLoader(data_dir=dest_dir)
    loader.load_all(downcast=True, ignore_missing=False, nrows=None)
    authenticity = loader.check_data_authenticity()
    print(authenticity["message"])
    if not authenticity["is_authentic"]:
        for tname, detail in authenticity["tables"].items():
            rows, cols = detail["actual_shape"]
            erows, ecols = detail["expected_min_shape"]
            status = "OK" if detail["authentic"] else "TOO SMALL"
            print(f"  - {tname}: {rows:,} x {cols} (expected >= {erows:,} x {ecols}) {status}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the real Home Credit Default Risk dataset from Kaggle."
    )
    parser.add_argument("--dest", type=str, default="data", help="Target directory for CSVs (default: data)")
    parser.add_argument("--force", action="store_true", help="Re-download even if all CSVs already exist")
    args = parser.parse_args()

    dest_dir = Path(args.dest)

    print("=" * 80)
    print("  HOME CREDIT DEFAULT RISK - REAL DATASET FETCHER")
    print("=" * 80)

    token = find_kaggle_token()
    if token:
        print(f"[OK] Kaggle token found: {token}")
        check_token_permissions(token)
    else:
        print(
            "[ERROR] kaggle.json not found.\n"
            "  1. Go to https://www.kaggle.com/settings and click 'Create New Token'\n"
            "     (this downloads kaggle.json).\n"
            "  2. Place it at ~/.kaggle/kaggle.json (or in the project root / server/).\n"
            "  3. Accept the competition rules at "
            f"https://www.kaggle.com/c/{COMPETITION_SLUG}/rules\n"
            "  4. Re-run this script."
        )
        sys.exit(1)

    if not ensure_kaggle_installed():
        sys.exit(1)

    download_and_extract(dest_dir, force=args.force)

    if validate_against_loader(dest_dir):
        print("\n[SUCCESS] Real dataset downloaded and validated - ready for preprocessing.")
        print("Next: python server/run_loader.py  (regenerates the data quality report)")
    else:
        print("\n[FAIL] Downloaded files do not match the real dataset scale.")
        sys.exit(1)


if __name__ == "__main__":
    main()
