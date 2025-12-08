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
    loss_file = CASE_A_DIR/ "onsite_lossesinGHz_afteradding_intrinsicloss.txt"
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


# Data directory for Case B
CASE_B_DIR = BASE_DIR / "data" / "case_B"

def load_case_B(use_compensated: bool = False):
    """
    Load spectra X and resonant frequencies y for Case B.

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
    loss_file = CASE_B_DIR / "finalized_resonant_freqs_wrt_ref_phase_using_fsr_and_puc_phases.txt"
    y = np.loadtxt(loss_file, delimiter=" ")
    y = y.reshape(-1)  # flatten to 1D

    # --- spectra ---
    # original spectra (in dB)
    spec_file = CASE_B_DIR / "full.txt"

    X_dB = np.loadtxt(spec_file, delimiter=" ")

    # convert from dB to linear scale
    X = 10.0 ** (X_dB / 10.0)

    # --- basic sanity check ---
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Mismatch between X and y: X has {X.shape[0]} rows, y has {y.shape[0]} entries."
        )

    return X, y

import numpy as np
from pathlib import Path

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parents[2]

# Data directory for each case
CASE_C_DIR = BASE_DIR / "data" / "case_C"


def load_case_C(use_compensated: bool = False):
    """
    Load spectra X and the two labels (ω_n, δ_n) for Case C.

    Case C has *two* target arrays:
        - resonant frequencies ω_n
        - onsite losses δ_n

    Both are flattened and combined into shape (N, 2).

    Parameters
    ----------
    use_compensated : bool
        If True, loads compensated spectra instead of full.txt.

    Returns
    -------
    X : np.ndarray
        Array of shape (n_samples, n_freq_bins) in linear units.
    y : np.ndarray
        Array of shape (n_samples, 2), columns = [omega, delta].
    """

    # --------- Load ω_n (resonant frequencies) ---------
    freq_file = CASE_C_DIR / "finalized_resonant_freqs_wrt_ref_phase_using_fsr_and_puc_phases.txt"
    datay0 = np.loadtxt(freq_file, delimiter=" ")
    datay0_flat = datay0.reshape(-1)

    # --------- Load δ_n (onsite losses) ---------
    delta_file =  CASE_C_DIR/ "random_oniste_losses_inGHz.txt"
    datay1 = np.loadtxt(delta_file, delimiter=" ")
    datay1_flat = datay1.reshape(-1)

    # --------- Combine labels ---------
    y = np.stack((datay0_flat, datay1_flat), axis=1)  # shape: (N, 2)

    # --------- Load spectra X ---------
    spec_file =  CASE_C_DIR / "full.txt"

    X_dB = np.loadtxt(spec_file, delimiter=" ")
    X = 10.0 ** (X_dB / 10.0)  # convert from dB to linear

    # --------- Sanity check ---------
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Mismatch: X has {X.shape[0]} rows but y has {y.shape[0]} samples."
        )

    return X, y

