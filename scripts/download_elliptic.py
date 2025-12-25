"""
Download and verify Elliptic Bitcoin Dataset.

Downloads the dataset from Kaggle and verifies integrity.
Requires: kaggle API credentials configured (~/.kaggle/kaggle.json)
"""

import os
import sys
import hashlib
import zipfile
from pathlib import Path
from typing import Optional
import argparse


# Expected MD5 checksums for data integrity
EXPECTED_MD5 = {
    'elliptic_txs_features.csv': None,  # Will be computed on first download
    'elliptic_txs_classes.csv': None,
    'elliptic_txs_edgelist.csv': None
}


def compute_md5(file_path: Path) -> str:
    """Compute MD5 checksum of file."""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def download_from_kaggle(output_dir: Path) -> bool:
    """
    Download Elliptic dataset from Kaggle.

    Parameters
    ----------
    output_dir : Path
        Directory to save downloaded files

    Returns
    -------
    bool
        True if download successful
    """
    try:
        import kaggle

        print("[INFO] Downloading Elliptic dataset from Kaggle...")
        print("       Dataset: ellipticco/elliptic-data-set")

        # Download dataset
        kaggle.api.dataset_download_files(
            'ellipticco/elliptic-data-set',
            path=str(output_dir),
            unzip=True
        )

        print("[OK] Download completed")
        return True

    except ImportError:
        print("[ERROR] Kaggle API not installed")
        print("        Install: pip install kaggle")
        print("        Configure: Place kaggle.json in ~/.kaggle/")
        print("        Get API key from: https://www.kaggle.com/settings/account")
        return False

    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def verify_dataset(data_dir: Path) -> bool:
    """
    Verify dataset integrity.

    Parameters
    ----------
    data_dir : Path
        Directory containing dataset files

    Returns
    -------
    bool
        True if verification passed
    """
    print("\n[INFO] Verifying dataset integrity...")

    expected_files = [
        'elliptic_txs_features.csv',
        'elliptic_txs_classes.csv',
        'elliptic_txs_edgelist.csv'
    ]

    all_passed = True

    for filename in expected_files:
        filepath = data_dir / filename

        # Check file exists
        if not filepath.exists():
            print(f"[ERROR] Missing file: {filename}")
            all_passed = False
            continue

        # Check file size
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"[OK] {filename}: {size_mb:.2f} MB")

        # Compute MD5
        md5 = compute_md5(filepath)
        print(f"     MD5: {md5}")

        # Verify against expected (if available)
        if EXPECTED_MD5[filename] is not None:
            if md5 != EXPECTED_MD5[filename]:
                print(f"[WARNING] MD5 mismatch for {filename}")
                print(f"          Expected: {EXPECTED_MD5[filename]}")
                print(f"          Got:      {md5}")
                all_passed = False
        else:
            print(f"     (No expected MD5 checksum available)")

    # Verify row counts (basic sanity check)
    import pandas as pd

    print("\n[INFO] Verifying row counts...")

    features_df = pd.read_csv(data_dir / 'elliptic_txs_features.csv', header=None)
    classes_df = pd.read_csv(data_dir / 'elliptic_txs_classes.csv')
    edges_df = pd.read_csv(data_dir / 'elliptic_txs_edgelist.csv')

    # Expected counts from paper
    expected_counts = {
        'features': 203769,
        'edges': 234355,
        'labeled': 46564  # 4545 illicit + 42019 licit
    }

    actual_counts = {
        'features': len(features_df),
        'edges': len(edges_df),
        'labeled': len(classes_df)
    }

    for key in expected_counts:
        expected = expected_counts[key]
        actual = actual_counts[key]

        if actual == expected:
            print(f"[OK] {key}: {actual:,} rows")
        else:
            print(f"[WARNING] {key}: Expected {expected:,}, got {actual:,}")
            all_passed = False

    # Verify feature count (should be 167 columns: txId + timestep + 166 features)
    if features_df.shape[1] == 167:
        print(f"[OK] features: 167 columns (txId + timestep + 165 features)")
    else:
        print(f"[ERROR] features: Expected 167 columns, got {features_df.shape[1]}")
        all_passed = False

    # Verify class distribution
    class_counts = classes_df['class'].value_counts()
    licit = class_counts.get('1', 0)
    illicit = class_counts.get('2', 0)

    print(f"\n[INFO] Class distribution:")
    print(f"       Licit (1):   {licit:,} ({licit/len(classes_df)*100:.1f}%)")
    print(f"       Illicit (2): {illicit:,} ({illicit/len(classes_df)*100:.1f}%)")
    print(f"       Imbalance ratio: {licit/illicit:.1f}:1")

    if all_passed:
        print("\n[OK] Dataset verification PASSED")
    else:
        print("\n[WARNING] Dataset verification completed with warnings")

    return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download and verify Elliptic Bitcoin Dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw/elliptic',
        help='Output directory for dataset (default: data/raw/elliptic)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download, only verify existing data'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Elliptic Bitcoin Dataset Download & Verification")
    print("=" * 70)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    # Download
    if not args.skip_download:
        success = download_from_kaggle(output_dir)
        if not success:
            print("\n[ERROR] Download failed. Exiting.")
            print("\nManual download instructions:")
            print("1. Go to: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
            print("2. Click 'Download' button")
            print(f"3. Extract to: {output_dir.absolute()}")
            sys.exit(1)

    # Verify
    verified = verify_dataset(output_dir)

    if verified:
        print("\n" + "=" * 70)
        print("SUCCESS: Dataset ready for use")
        print("=" * 70)
        print(f"\nNext steps:")
        print(f"1. Run feature engineering: python scripts/preprocess_data.py")
        print(f"2. Train baseline models: python scripts/train_baseline.py")
        print(f"3. Train CHRONOS-Net: python scripts/train_chronos.py")
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("WARNING: Dataset verification completed with warnings")
        print("=" * 70)
        print("You may proceed, but data integrity cannot be guaranteed.")
        sys.exit(1)


if __name__ == '__main__':
    main()
