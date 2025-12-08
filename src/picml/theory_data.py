import numpy as np
from pathlib import Path
from scipy import linalg


# ----------------------------
#  Low-level physics helpers
# ----------------------------

def eigensys_A(v: np.ndarray, t: float = 0.82) -> np.ndarray:
    """
    Build the non-Hermitian Hamiltonian for Case A and return it.

    H = -i diag(v) + t (|n><n+1| + |n+1><n|), multiplied by 2π.

    Parameters
    ----------
    v : np.ndarray
        Onsite losses, shape (N,).
    t : float
        Nearest-neighbour coupling.

    Returns
    -------
    H : np.ndarray
        Complex Hamiltonian matrix of shape (N, N).
    """
    v = np.asarray(v)
    N = len(v)
    H1 = np.diag(-1j * v) + t * np.eye(N, k=1) + t * np.eye(N, k=-1)
    return 2.0 * np.pi * H1


def eigensys_B_C(v: np.ndarray, vtilde: np.ndarray, t: float = 0.82) -> np.ndarray:
    """
    Build the non-Hermitian Hamiltonian for Cases B/C.

    H = diag(v) - i diag(vtilde) + t (|n><n+1| + |n+1><n|), multiplied by 2π.
    """
    v = np.asarray(v)
    vtilde = np.asarray(vtilde)
    N = len(v)
    H1 = np.diag(v - 1j * vtilde) + t * np.eye(N, k=1) + t * np.eye(N, k=-1)
    return 2.0 * np.pi * H1


def _orthomatrix(vl_left: np.ndarray, vl_right: np.ndarray) -> np.ndarray:
    """
    Overlap matrix between left and right eigenvectors: M = <L|R>.
    """
    return np.conj(vl_left.T) @ vl_right


def new_biorthogonal_basis(vl_left: np.ndarray, vl_right: np.ndarray, N: int):
    """
    Construct a biorthogonal basis from left and right eigenvectors using LU
    factorization of the overlap matrix.

    Returns
    -------
    vlp_left, vlp_right : np.ndarray
        Biorthogonalized left and right eigenvectors, shape (N, N) each.
    """
    M = _orthomatrix(vl_left, vl_right)
    p, l, u = linalg.lu(M)
    linv = linalg.inv(p @ l)
    uinv = linalg.inv(u)

    vlp_left = linv @ np.conj(vl_left.T)
    vlp_right = vl_right @ uinv

    M1 = vlp_left @ vlp_right
    # Optional sanity check:
    # is_biorthogonal = np.allclose(M1, np.eye(N), atol=1e-10)

    # Return in the same "left/right" orientation as your original code
    return np.conj(vlp_left.T), vlp_right


def gpower(
    eigenvalues: np.ndarray,
    vl_left: np.ndarray,
    vl_right: np.ndarray,
    omega_drive: float,
    gamma: float = 0.1,
) -> np.ndarray:
    """
    Compute the spectral power at drive frequency omega_drive using the
    Green-function-like expression.

    pw_n = | sum_m <n|L_m> <R_m|n> / (omega_drive - w_m + i gamma) |^2

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues w_m, shape (N,).
    vl_left : np.ndarray
        Left eigenvectors in biorthogonal basis, shape (N, N).
    vl_right : np.ndarray
        Right eigenvectors in biorthogonal basis, shape (N, N).
    omega_drive : float
        Driving angular frequency.
    gamma : float
        Broadening parameter (imaginary frequency shift).

    Returns
    -------
    pw : np.ndarray
        Power at each site, shape (N,).
    """
    # eigenvalues: (N,)
    # vl_left, vl_right: (N, N)
    denom = omega_drive - eigenvalues + 1j * gamma  # shape (N,)
    # (conj(vl_left) * vl_right) has shape (N, N); divide row-wise by denom
    frac = np.conj(vl_left) * vl_right / denom[np.newaxis, :]
    pw = np.square(np.abs(np.sum(frac, axis=1)))
    return pw

def generate_case_A_model2_spectra(
    predicted_losses_train: np.ndarray,
    true_losses_train: np.ndarray,
    predicted_losses_test: np.ndarray,
    frequencies: np.ndarray,
    gamma: float = 0.1,
    coupling: float = 0.82,
):
    """
    Generate theoretical spectra (X) for model 2 of Case A from predicted and
    true onsite losses and a list of drive frequencies.

    This is a cleaned version of the notebook code that builds:
        - X1_train (from predicted losses)
        - true_X1_train (from true losses)
        - X1_test (from predicted test losses)

    Parameters
    ----------
    predicted_losses_train : np.ndarray
        Array of shape (n_train_groups, N), grouped predicted losses for training.
    true_losses_train : np.ndarray
        Array of shape (n_train_groups, N), true losses for training.
    predicted_losses_test : np.ndarray
        Array of shape (n_test_groups, N), grouped predicted losses for test.
    frequencies : np.ndarray
        1D array of angular frequencies (already multiplied by 2π if desired).
    gamma : float
        Broadening parameter used in gpower().
    coupling : float
        Nearest-neighbour coupling t in the Hamiltonian.

    Returns
    -------
    X1_train : np.ndarray
        Theoretical spectra from predicted_losses_train, shape
        (n_train_groups * N, n_freqs).
    true_X1_train : np.ndarray
        Theoretical spectra from true_losses_train, shape
        (n_train_groups * N, n_freqs).
    X1_test : np.ndarray
        Theoretical spectra from predicted_losses_test, shape
        (n_test_groups * N, n_freqs).
    """
    predicted_losses_train = np.asarray(predicted_losses_train)
    true_losses_train = np.asarray(true_losses_train)
    predicted_losses_test = np.asarray(predicted_losses_test)
    frequencies = np.asarray(frequencies)

    n_train_groups, N = predicted_losses_train.shape
    n_test_groups, N_test = predicted_losses_test.shape
    assert N == N_test, "Train and test group size (number of sites) must match."

    n_freqs = frequencies.shape[0]

    # Allocate arrays
    X1_train = np.zeros((n_train_groups * N, n_freqs), dtype=np.float64)
    true_X1_train = np.zeros((n_train_groups * N, n_freqs), dtype=np.float64)
    X1_test = np.zeros((n_test_groups * N, n_freqs), dtype=np.float64)

    # Pre-enumerate frequencies for looping
    frequencies_num = list(enumerate(frequencies))

    # --- Training data: predicted vs true ---
    for i_group in range(n_train_groups):
        v_pred = predicted_losses_train[i_group, :]
        v_true = true_losses_train[i_group, :]

        # Hamiltonians
        Ham_pred = eigensys_A(v_pred, t=coupling)
        w_pred, vl_left_pred, vl_right_pred = linalg.eig(Ham_pred, left=True, right=True)
        vlp_left_pred, vlp_right_pred = new_biorthogonal_basis(
            vl_left_pred, vl_right_pred, N
        )

        Ham_true = eigensys_A(v_true, t=coupling)
        w_true, vl_left_true, vl_right_true = linalg.eig(
            Ham_true, left=True, right=True
        )
        vlp_left_true, vlp_right_true = new_biorthogonal_basis(
            vl_left_true, vl_right_true, N
        )

        # Fill spectra over all frequencies
        row_slice = slice(i_group * N, (i_group + 1) * N)

        for i_freq, omega in frequencies_num:
            X1_train[row_slice, i_freq] = gpower(
                w_pred, vlp_left_pred, vlp_right_pred, omega, gamma=gamma
            )
            true_X1_train[row_slice, i_freq] = gpower(
                w_true, vlp_left_true, vlp_right_true, omega, gamma=gamma
            )

    # --- Test data: predicted only ---
    for i_group in range(n_test_groups):
        v_pred_test = predicted_losses_test[i_group, :]

        Ham_test = eigensys_A(v_pred_test, t=coupling)
        w_test, vl_left_test, vl_right_test = linalg.eig(
            Ham_test, left=True, right=True
        )
        vlp_left_test, vlp_right_test = new_biorthogonal_basis(
            vl_left_test, vl_right_test, N
        )

        row_slice = slice(i_group * N, (i_group + 1) * N)
        for i_freq, omega in frequencies_num:
            X1_test[row_slice, i_freq] = gpower(
                w_test, vlp_left_test, vlp_right_test, omega, gamma=gamma
            )

    return X1_train, true_X1_train, X1_test

def generate_case_B_model2_spectra(
    predicted_v_train: np.ndarray,
    predicted_vtilde_train: np.ndarray,
    true_v_train: np.ndarray,
    true_vtilde_train: np.ndarray,
    predicted_v_test: np.ndarray,
    predicted_vtilde_test: np.ndarray,
    frequencies: np.ndarray,
    gamma: float = 0.1,
    coupling: float = 0.82,
):
    """
    Generate theoretical spectra (X) for model 2 of Case B from predicted and
    true parameters (v, vtilde) and a list of drive frequencies.

    Parameters
    ----------
    predicted_v_train : np.ndarray
        Shape (n_train_groups, N). Real part parameters for training (e.g. ω or similar).
    predicted_vtilde_train : np.ndarray
        Shape (n_train_groups, N). Imaginary part parameters (e.g. losses) for training.
    true_v_train : np.ndarray
        Shape (n_train_groups, N). True real-part parameters for training.
    true_vtilde_train : np.ndarray
        Shape (n_train_groups, N). True imaginary-part parameters for training.
    predicted_v_test : np.ndarray
        Shape (n_test_groups, N). Predicted real-part parameters for test.
    predicted_vtilde_test : np.ndarray
        Shape (n_test_groups, N). Predicted imaginary-part parameters for test.
    frequencies : np.ndarray
        1D array of angular drive frequencies.
    gamma : float
        Broadening parameter used in gpower().
    coupling : float
        Nearest-neighbour coupling t in the Hamiltonian.

    Returns
    -------
    X_train_pred : np.ndarray
        Theoretical spectra from predicted (v, vtilde) for training,
        shape (n_train_groups * N, n_freqs).
    X_train_true : np.ndarray
        Theoretical spectra from true (v, vtilde) for training,
        shape (n_train_groups * N, n_freqs).
    X_test_pred : np.ndarray
        Theoretical spectra from predicted (v, vtilde) for test,
        shape (n_test_groups * N, n_freqs).
    """
    predicted_v_train = np.asarray(predicted_v_train)
    predicted_vtilde_train = np.asarray(predicted_vtilde_train)
    true_v_train = np.asarray(true_v_train)
    true_vtilde_train = np.asarray(true_vtilde_train)
    predicted_v_test = np.asarray(predicted_v_test)
    predicted_vtilde_test = np.asarray(predicted_vtilde_test)
    frequencies = np.asarray(frequencies)

    n_train_groups, N = predicted_v_train.shape
    n_test_groups, N_test = predicted_v_test.shape
    assert N == N_test, "Train and test group size (number of sites) must match."

    n_freqs = frequencies.shape[0]

    X_train_pred = np.zeros((n_train_groups * N, n_freqs), dtype=np.float64)
    X_train_true = np.zeros((n_train_groups * N, n_freqs), dtype=np.float64)
    X_test_pred = np.zeros((n_test_groups * N, n_freqs), dtype=np.float64)

    frequencies_num = list(enumerate(frequencies))

    # ---- Training: predicted vs true ----
    for i_group in range(n_train_groups):
        v_pred = predicted_v_train[i_group, :]
        vtilde_pred = predicted_vtilde_train[i_group, :]
        v_true = true_v_train[i_group, :]
        vtilde_true = true_vtilde_train[i_group, :]

        # Predicted Hamiltonian and biorthogonal basis
        Ham_pred = eigensys_B_C(v_pred, vtilde_pred, t=coupling)
        w_pred, vl_left_pred, vl_right_pred = linalg.eig(
            Ham_pred, left=True, right=True
        )
        vlp_left_pred, vlp_right_pred = new_biorthogonal_basis(
            vl_left_pred, vl_right_pred, N
        )

        # True Hamiltonian and basis
        Ham_true = eigensys_B_C(v_true, vtilde_true, t=coupling)
        w_true, vl_left_true, vl_right_true = linalg.eig(
            Ham_true, left=True, right=True
        )
        vlp_left_true, vlp_right_true = new_biorthogonal_basis(
            vl_left_true, vl_right_true, N
        )

        row_slice = slice(i_group * N, (i_group + 1) * N)

        for i_freq, omega in frequencies_num:
            X_train_pred[row_slice, i_freq] = gpower(
                w_pred, vlp_left_pred, vlp_right_pred, omega, gamma=gamma
            )
            X_train_true[row_slice, i_freq] = gpower(
                w_true, vlp_left_true, vlp_right_true, omega, gamma=gamma
            )

    # ---- Test: predicted only ----
    for i_group in range(n_test_groups):
        v_pred_test = predicted_v_test[i_group, :]
        vtilde_pred_test = predicted_vtilde_test[i_group, :]

        Ham_test = eigensys_B_C(v_pred_test, vtilde_pred_test, t=coupling)
        w_test, vl_left_test, vl_right_test = linalg.eig(
            Ham_test, left=True, right=True
        )
        vlp_left_test, vlp_right_test = new_biorthogonal_basis(
            vl_left_test, vl_right_test, N
        )

        row_slice = slice(i_group * N, (i_group + 1) * N)
        for i_freq, omega in frequencies_num:
            X_test_pred[row_slice, i_freq] = gpower(
                w_test, vlp_left_test, vlp_right_test, omega, gamma=gamma
            )

    return X_train_pred, X_train_true, X_test_pred

def generate_case_C_model2_spectra(
    predicted_v_train: np.ndarray,
    predicted_vtilde_train: np.ndarray,
    true_v_train: np.ndarray,
    true_vtilde_train: np.ndarray,
    predicted_v_test: np.ndarray,
    predicted_vtilde_test: np.ndarray,
    frequencies: np.ndarray,
    gamma: float = 0.1,
    coupling: float = 0.82,
):
    """
    Generate theoretical spectra (X) for model 2 of Case C from predicted and
    true parameters (v, vtilde) and a list of drive frequencies.

    Parameters
    ----------
    predicted_v_train : np.ndarray
        Shape (n_train_groups, N). Real-part parameters for training (e.g. ω_n).
    predicted_vtilde_train : np.ndarray
        Shape (n_train_groups, N). Imag-part parameters for training (e.g. δ_n).
    true_v_train : np.ndarray
        Shape (n_train_groups, N). True real-part parameters for training.
    true_vtilde_train : np.ndarray
        Shape (n_train_groups, N). True imag-part parameters for training.
    predicted_v_test : np.ndarray
        Shape (n_test_groups, N). Predicted real-part parameters for test.
    predicted_vtilde_test : np.ndarray
        Shape (n_test_groups, N). Predicted imag-part parameters for test.
    frequencies : np.ndarray
        1D array of angular drive frequencies.
    gamma : float
        Broadening parameter used in gpower().
    coupling : float
        Nearest-neighbour coupling t in the Hamiltonian.

    Returns
    -------
    X_train_pred : np.ndarray
        Theoretical spectra from predicted (v, vtilde) for training,
        shape (n_train_groups * N, n_freqs).
    X_train_true : np.ndarray
        Theoretical spectra from true (v, vtilde) for training,
        shape (n_train_groups * N, n_freqs).
    X_test_pred : np.ndarray
        Theoretical spectra from predicted (v, vtilde) for test,
        shape (n_test_groups * N, n_freqs).
    """
    predicted_v_train = np.asarray(predicted_v_train)
    predicted_vtilde_train = np.asarray(predicted_vtilde_train)
    true_v_train = np.asarray(true_v_train)
    true_vtilde_train = np.asarray(true_vtilde_train)
    predicted_v_test = np.asarray(predicted_v_test)
    predicted_vtilde_test = np.asarray(predicted_vtilde_test)
    frequencies = np.asarray(frequencies)

    n_train_groups, N = predicted_v_train.shape
    n_test_groups, N_test = predicted_v_test.shape
    assert N == N_test, "Train and test group size (number of sites) must match."

    n_freqs = frequencies.shape[0]

    X_train_pred = np.zeros((n_train_groups * N, n_freqs), dtype=np.float64)
    X_train_true = np.zeros((n_train_groups * N, n_freqs), dtype=np.float64)
    X_test_pred = np.zeros((n_test_groups * N, n_freqs), dtype=np.float64)

    frequencies_num = list(enumerate(frequencies))

    # ---- Training: predicted vs true ----
    for i_group in range(n_train_groups):
        v_pred = predicted_v_train[i_group, :]
        vtilde_pred = predicted_vtilde_train[i_group, :]
        v_true = true_v_train[i_group, :]
        vtilde_true = true_vtilde_train[i_group, :]

        Ham_pred = eigensys_B_C(v_pred, vtilde_pred, t=coupling)
        w_pred, vl_left_pred, vl_right_pred = linalg.eig(
            Ham_pred, left=True, right=True
        )
        vlp_left_pred, vlp_right_pred = new_biorthogonal_basis(
            vl_left_pred, vl_right_pred, N
        )

        Ham_true = eigensys_B_C(v_true, vtilde_true, t=coupling)
        w_true, vl_left_true, vl_right_true = linalg.eig(
            Ham_true, left=True, right=True
        )
        vlp_left_true, vlp_right_true = new_biorthogonal_basis(
            vl_left_true, vl_right_true, N
        )

        row_slice = slice(i_group * N, (i_group + 1) * N)
        for i_freq, omega in frequencies_num:
            X_train_pred[row_slice, i_freq] = gpower(
                w_pred, vlp_left_pred, vlp_right_pred, omega, gamma=gamma
            )
            X_train_true[row_slice, i_freq] = gpower(
                w_true, vlp_left_true, vlp_right_true, omega, gamma=gamma
            )

    # ---- Test: predicted only ----
    for i_group in range(n_test_groups):
        v_pred_test = predicted_v_test[i_group, :]
        vtilde_pred_test = predicted_vtilde_test[i_group, :]

        Ham_test = eigensys_B_C(v_pred_test, vtilde_pred_test, t=coupling)
        w_test, vl_left_test, vl_right_test = linalg.eig(
            Ham_test, left=True, right=True
        )
        vlp_left_test, vlp_right_test = new_biorthogonal_basis(
            vl_left_test, vl_right_test, N
        )

        row_slice = slice(i_group * N, (i_group + 1) * N)
        for i_freq, omega in frequencies_num:
            X_test_pred[row_slice, i_freq] = gpower(
                w_test, vlp_left_test, vlp_right_test, omega, gamma=gamma
            )

    return X_train_pred, X_train_true, X_test_pred
