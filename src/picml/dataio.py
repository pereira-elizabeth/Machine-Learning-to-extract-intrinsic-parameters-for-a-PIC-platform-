# picml: ML utilities for parameter inference on PIC spectra
import numpy as np
from pathlib import Path

# Base directory: .../your-repo/
BASE_DIR = Path(__file__).resolve().parents[2]

# Data directory for Case A
CASE_A_DIR = BASE_DIR / "data" / "case_A"


def load_case_A(use_compensated: bool = False):
    """
    Load spectra X and onsite losses y for Case A.

    Parameters
    ----------
    use_compensated : bool, optional
        If True, load compensated spectra (e.g. 'compensated_spectral_data.txt')
        instead of 'full.txt'. Adjust filenames below to match your data.

    Returns
    -------
    X : np.ndarray
        2D array of shape (n_samples, n_freq_bins), spectra in linear units.
    y : np.ndarray
        1D array of shape (n_samples,), onsite losses in GHz.
    """
    # --- onsite losses ---
    loss_file = CASE_A_DIR / "onsite_lossesinGHz_afteradding_intrinsicloss.txt"
    y = np.loadtxt(loss_file, delimiter=" ")
    y = y.reshape(-1)  # flatten to 1D

    # --- spectra ---
    # original spectra (in dB)
    spec_file = CASE_A_DIR / "full.txt"

    X_dB = np.loadtxt(spec_file, delimiter=" ")

    # convert from dB to linear scale
    X = 10.0 ** (X_dB / 10.0)

    # --- basic sanity check ---
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Mismatch between X and y: X has {X.shape[0]} rows, y has {y.shape[0]} entries."
        )

    return X, y
