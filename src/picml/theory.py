import numpy as np
from pathlib import Path
from scipy import linalg



# ----------------------------
#  Low-level physics helpers
# ----------------------------
def load_frequencies(path, scale_2pi: bool = True) -> np.ndarray:
    freqs = np.loadtxt(path, delimiter=" ")
    return (2.0 * np.pi * freqs) if scale_2pi else freqs

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
    true_losses_test: np.ndarray,
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
    X1_test_true = np.zeros((n_test_groups * N, n_freqs), dtype=np.float64)


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
        v_true_test = true_losses_test[i_group, :]
    
        # predicted
        Ham_pred = eigensys_A(v_pred_test, t=coupling)
        w_pred, vlL_pred, vlR_pred = linalg.eig(Ham_pred, left=True, right=True)
        vlpL_pred, vlpR_pred = new_biorthogonal_basis(vlL_pred, vlR_pred, N)
    
        # true
        Ham_true = eigensys_A(v_true_test, t=coupling)
        w_true, vlL_true, vlR_true = linalg.eig(Ham_true, left=True, right=True)
        vlpL_true, vlpR_true = new_biorthogonal_basis(vlL_true, vlR_true, N)
    
        row_slice = slice(i_group * N, (i_group + 1) * N)
    
        for i_freq, omega in frequencies_num:
            X1_test[row_slice, i_freq] = gpower(
                w_pred, vlpL_pred, vlpR_pred, omega, gamma=gamma
            )
            X1_test_true[row_slice, i_freq] = gpower(
                w_true, vlpL_true, vlpR_true, omega, gamma=gamma
            )


    return X1_train, true_X1_train, X1_test, X1_test_true


def generate_case_B_model2_spectra(
    y_combined_true: np.ndarray,
    y_combined_pred: np.ndarray,
    frequencies: np.ndarray,
    gamma: float = 0.1,
    coupling: float = 0.82,
):
    """
    Generate theoretical spectra (X) for model 2 of Case B from predicted and
    true parameters (v, vtilde) and a list of drive frequencies.

    Returns
    -------
    X_predicted : np.ndarray
        Theoretical spectra from predicted (v, vtilde) for training,
        shape (n_train_groups * N, n_freqs).
    X_true : np.ndarray
        Theoretical spectra from true (v, vtilde) for training,
        shape (n_train_groups * N, n_freqs).
    """
    frequencies_num = list(enumerate(frequencies))
    FSRs_ang = np.array([81.06753754801633, 83.41732126685432, 83.41764401355516, 82.24227030736816, 
                 82.24290670919115, 82.24258850750248, 82.24258850750248, 83.41764401355516])
    FSRs_scaling = FSRs_ang/(2.*np.pi)/4.
    l = [0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15]
    Losses = l * FSRs_scaling
    X_true = np.zeros((y_combined_true.shape[0] *y_combined_true.shape[1] , len(frequencies)), dtype=np.float64)
    X_predicted = np.zeros((y_combined_pred.shape[0] *y_combined_pred.shape[1] , len(frequencies)), dtype=np.float64)
    # ---- Training: predicted vs true ----
    for i1 in range(y_combined_true.shape[0]):
        Ham_true = eigensys_B_C(y_combined_true[i1,:],Losses)
        w_true, vl1_true, vl2_true = linalg.eig(Ham_true, left=True, right=True)
        vlp1_true, vlp2_true = new_biorthogonal_basis(vl1_true, vl2_true,y_combined_true.shape[1])
    
        Ham_predicted = eigensys_B_C(y_combined_pred[i1,:],Losses)
        w_predicted, vl1_predicted, vl2_predicted = linalg.eig(Ham_predicted, left=True, right=True)
        vlp1_predicted, vlp2_predicted = new_biorthogonal_basis(vl1_predicted, vl2_predicted,y_combined_pred.shape[1])
    
        for i2, freq in frequencies_num:
           X_true[i1 * y_combined_true.shape[1]: (i1 + 1) * y_combined_true.shape[1], i2] = gpower(w_true, vlp1_true, vlp2_true, freq,gamma=0.5)   
           X_predicted[i1 * y_combined_pred.shape[1]: (i1 + 1) * y_combined_pred.shape[1], i2] =gpower(w_predicted, vlp1_predicted, vlp2_predicted, freq,gamma=0.5)     

    return X_true, X_predicted

def generate_case_C_model2_spectra(
    y_true_l1: np.ndarray,
    y_pred_l1: np.ndarray,
    y_true_f1: np.ndarray,
    y_pred_f1: np.ndarray,
    frequencies: np.ndarray,
    gamma: float = 0.1,
    coupling: float = 0.82,
):
    """
    Generate theoretical spectra (X) for model 2 of Case C from predicted and
    true parameters (v, vtilde) and a list of drive frequencies.
    """

    frequencies_num = list(enumerate(frequencies))

    X_true = np.zeros((y_true_l1.shape[0] *y_true_l1.shape[1] , len(frequencies)), dtype=np.float64)
    X_predicted = np.zeros((y_pred_l1.shape[0] *y_pred_l1.shape[1] , len(frequencies)), dtype=np.float64)

    for i1 in range(y_true_l1.shape[0]):
        Ham_true = eigensys_B_C(y_true_f1[i1,:],y_true_l1[i1,:])
        w_true, vl1_true, vl2_true = linalg.eig(Ham_true, left=True, right=True)
        vlp1_true, vlp2_true = new_biorthogonal_basis(vl1_true, vl2_true,y_true_l1.shape[1])
    
        Ham_predicted = eigensys_B_C(y_pred_f1[i1,:],y_pred_l1[i1,:])
        w_predicted, vl1_predicted, vl2_predicted = linalg.eig(Ham_predicted, left=True, right=True)
        vlp1_predicted, vlp2_predicted = new_biorthogonal_basis(vl1_predicted, vl2_predicted,y_pred_l1.shape[1])
    
        for i2, freq in frequencies_num:
            X_true[i1 * y_true_l1.shape[1]: (i1 + 1) * y_true_l1.shape[1], i2] = gpower(w_true, vlp1_true, vlp2_true, freq,gamma=0.5)   
            X_predicted[i1 * y_pred_l1.shape[1]: (i1 + 1) * y_pred_l1.shape[1], i2] = gpower(w_predicted, vlp1_predicted, vlp2_predicted, freq,gamma=0.5)     

    return X_true, X_predicted

def select_datay_ring_major_order(datay, X, ref_freqs, epsilon):
    datay = np.array(datay.reshape(-1,8))
    ref_freqs = np.array(ref_freqs)
    epsilon = np.array(epsilon)
    
    n_datapoints, n_rings = datay.shape
    assert X.shape[0] == n_datapoints * n_rings
    assert ref_freqs.shape[0] == epsilon.shape[0] == n_rings

    chosendatay = []
    chosenX = []
    chosenY = []

    for i in range(n_datapoints):          # loop over each row
        for j in range(n_rings):           # loop over each ring
            val = datay[i, j]
            if abs(val - ref_freqs[j]) <= epsilon[j]:
                idx = i * n_rings + j
                chosendatay.append(val)
                chosenX.append(X[idx])

    return np.array(chosendatay), np.array(chosenX)    

def filter_by_puc_phase_ring_major(datay0, datay1, X, ref_puc_phases, epsilon):
    datay0 = np.array(datay0)
    datay1 = np.array(datay1)
    ref_puc_phases = np.array(ref_puc_phases)
    epsilon = np.array(epsilon)

    n_datapoints, n_rings = datay0.shape
    assert datay1.shape == datay0.shape
    assert X.shape[0] == n_datapoints * n_rings
    assert ref_puc_phases.shape[0] == epsilon.shape[0] == n_rings

    chosendatay0 = []
    chosendatay1 = []
    chosenX = []

    for i in range(n_datapoints):
        for j in range(n_rings):
            if abs(datay0[i, j] - ref_puc_phases[j]) <= epsilon[j]:
                idx = i * n_rings + j
                chosendatay0.append(datay0[i, j])
                chosendatay1.append(datay1[i, j])
                chosenX.append(X[idx])

    return (
        np.array(chosendatay0),
        np.array(chosendatay1),
        np.array(chosenX)
    )
    
