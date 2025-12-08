import numpy as np


def filter_by_puc_phase_ring_major(
    datay0,
    datay1,
    X,
    ref_puc_phases,
    epsilon,
):
    """
    Filter datapoints ring-by-ring based on proximity to reference PUC phases.

    Parameters
    ----------
    datay0 : array-like, shape (n_datapoints, n_rings)
        PUC phases or similar per-sample, per-ring quantity.
    datay1 : array-like, shape (n_datapoints, n_rings)
        Second label array aligned with datay0 (e.g. onsite losses).
    X : array-like, shape (n_datapoints * n_rings, n_features)
        Spectral data arranged in ring-major order: for each sample i,
        the rows [i*n_rings : (i+1)*n_rings] correspond to rings j=0..n_rings-1.
    ref_puc_phases : array-like, shape (n_rings,)
        Reference PUC phase per ring.
    epsilon : array-like, shape (n_rings,)
        Tolerance per ring. Ring j is accepted if
        |datay0[i, j] - ref_puc_phases[j]| <= epsilon[j].

    Returns
    -------
    chosen_datay0 : np.ndarray, shape (n_selected,)
        Filtered values from datay0.
    chosen_datay1 : np.ndarray, shape (n_selected,)
        Corresponding filtered values from datay1.
    chosen_X : np.ndarray, shape (n_selected, n_features)
        Corresponding filtered spectral rows.
    """
    datay0 = np.asarray(datay0)
    datay1 = np.asarray(datay1)
    X = np.asarray(X)
    ref_puc_phases = np.asarray(ref_puc_phases)
    epsilon = np.asarray(epsilon)

    n_datapoints, n_rings = datay0.shape
    assert datay1.shape == datay0.shape, "datay0 and datay1 must have same shape"
    assert X.shape[0] == n_datapoints * n_rings, "X must be ring-major stacked"
    assert ref_puc_phases.shape[0] == n_rings, "ref_puc_phases must have length n_rings"
    assert epsilon.shape[0] == n_rings, "epsilon must have length n_rings"

    chosen_datay0 = []
    chosen_datay1 = []
    chosen_X = []

    for i in range(n_datapoints):
        for j in range(n_rings):
            if abs(datay0[i, j] - ref_puc_phases[j]) <= epsilon[j]:
                idx = i * n_rings + j
                chosen_datay0.append(datay0[i, j])
                chosen_datay1.append(datay1[i, j])
                chosen_X.append(X[idx])

    return (
        np.asarray(chosen_datay0),
        np.asarray(chosen_datay1),
        np.asarray(chosen_X),
    )


def fidelity_imag(y_pred, y_true):
    """
    Compute a 'fidelity' between two sets of imaginary components using a
    normalised covariance (essentially a correlation-like measure).

    Parameters
    ----------
    y_pred : array-like
        Predicted imaginary values.
    y_true : array-like
        True imaginary values.

    Returns
    -------
    F_imag : float
        Fidelity / correlation-like scalar in [-1, 1] (if well-behaved).
    """
    ai = np.asarray(y_pred, dtype=float)
    bi = np.asarray(y_true, dtype=float)

    mean_ai = np.mean(ai)
    mean_bi = np.mean(bi)

    num = np.mean(ai * bi) - mean_ai * mean_bi
    var_ai = np.mean(ai**2) - mean_ai**2
    var_bi = np.mean(bi**2) - mean_bi**2

    denom = (var_ai * var_bi) ** 0.5
    if denom == 0:
        return np.nan  # or 0.0, depending on what you prefer

    return num / denom


def distance_label(a, b):
    """
    Assign integer labels based on distance from the origin in (a, b) space.

    For each pair (a[i], b[i]), compute distance d_i = sqrt(a[i]^2 + b[i]^2),
    sort by increasing d_i, and assign rank 1 to the closest, 2 to the next, etc.

    Parameters
    ----------
    a : array-like
        First coordinate array.
    b : array-like
        Second coordinate array.

    Returns
    -------
    c : np.ndarray, shape (N,)
        Rank labels: c[i] is the rank of (a[i], b[i]) by distance from origin.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    dist = np.sqrt(a**2 + b**2)
    order = np.argsort(dist)

    c = np.empty_like(order)
    c[order] = np.arange(1, len(dist) + 1)
    return c


def sort_by_peak_frequency(z, S):
    """
    Sort spectra by peak frequency, breaking ties by peak amplitude.

    Parameters
    ----------
    z : array-like, shape (N,)
        Label/index array associated with each spectrum.
    S : array-like, shape (N, n_freqs)
        Spectral intensities; each row is a spectrum.

    Returns
    -------
    z_sorted : np.ndarray, shape (N,)
        z reordered according to sorting.
    S_sorted : np.ndarray, shape (N, n_freqs)
        S reordered according to sorting.
    order : np.ndarray, shape (N,)
        Indices such that z_sorted = z[order], S_sorted = S[order, :].
    """
    S = np.asarray(S)
    z = np.asarray(z)

    peak_idx = np.argmax(S, axis=1)
    peak_amp = S[np.arange(S.shape[0]), peak_idx]

    # lexsort sorts by last key first; here:
    #   primary key: peak_idx (frequency)
    #   secondary key: -peak_amp (stronger peak earlier)
    order = np.lexsort((-peak_amp, peak_idx))

    return z[order], S[order, :], order


