"""
Minimal test for Case A data loading.

This script checks that:
- load_case_A runs without error
- X and y have compatible shapes
- data looks numerically sane (no NaNs, no empty arrays)

Run with:
    python tests/test_load_case_A.py
"""

import sys
from pathlib import Path
import numpy as np

# --------------------------------------------------
# Make src/ importable
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]   # repo root
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from picml.dataio import load_case_A


def main():
    print("Loading Case A data...")

    X, y = load_case_A(use_compensated=False)

    print("Loaded successfully.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {np.asarray(y).shape}")

    # ----------------------------
    # Basic sanity checks
    # ----------------------------
    assert X.ndim == 2, "X should be a 2D array (samples × features)"
    assert len(y) == X.shape[0], (
        "Number of labels must match number of spectra"
    )

    assert not np.isnan(X).any(), "X contains NaNs"
    assert not np.isnan(y).any(), "y contains NaNs"

    assert X.shape[0] > 0, "X is empty"
    assert X.shape[1] > 0, "X has zero features"

    print("✔ Basic sanity checks passed.")

    # Optional: print value ranges (useful for debugging)
    print(f"X min/max: {X.min():.3e} / {X.max():.3e}")
    print(f"y min/max: {np.min(y):.3e} / {np.max(y):.3e}")

    print("✅ Case A data loader looks OK.")


if __name__ == "__main__":
    main()




















