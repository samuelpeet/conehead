# Spectral information for a 6MV Varian Clinac.
#
# From article:
# E S M Ali and D W O Rogers 2012 Phys. Med. Biol. 57 31
import numpy as np
from conehead.nist import mu_W, mu_Al
from scipy.interpolate import interp1d


def weights(E):
    """ Return spectrum weights for a Varian Clinac 6MV linac.

    From article E S M Ali and D W O Rogers 2012 Phys. Med. Biol. 57 31, 
    corresponding to the 6MV Varian Clinac in table 5.

    Parameters
    ----------
    E : ndarray
        Array of energies (units of MV)

    Returns
    -------
    ndarray
        Array of spectrum weights. The array is normalised if it contains more
        than one value.
    """
    # Equation 12 of Table 2.
    C_1 = 1.222
    C_2 = 5.147
    C_3 = -1.186
    E_e = 5.76
    psi_thin = (1 + C_3 * E / E_e + np.power(E / E_e, 2)) * \
               (np.log(E_e * (E_e - E) / E + 1.65) - 0.5)
    psi = psi_thin * np.exp(-mu_W(E) * np.power(C_1, 2) - mu_Al(E) *
                            np.power(C_2, 2))

    # Set negative results to zero
    if isinstance(psi, np.ndarray):
        psi[psi < 0] = 0
        N = np.trapz(psi, x=E)
        psi /= N
    else:
        # Probably just a single float
        if psi < 0 or np.isnan(psi):
            psi = 0
        psi = np.array(psi)

    return psi


def weights_sheikh_bagheri(E):
    """ Return spectrum weights for a Varian Clinac 6MV linac.

    From article D Sheikh-Bagheri and D W O Rogers 2002 Med. Phys. 29 3, 
    corresponding to the 6MV Varian Clinac in table 5.

    Parameters
    ----------
    E : ndarray
        Array of energies (units of MV)

    Returns
    -------
    ndarray
        Array of spectrum weights.
    """
    w = np.array([[0.250, 0.214e-4],
                  [0.500, 0.126e-3],
                  [0.750, 0.131e-3],
                  [1.000, 0.114e-3],
                  [1.250, 0.976e-4],
                  [1.500, 0.836e-4],
                  [1.750, 0.725e-4],
                  [2.000, 0.623e-4],
                  [2.250, 0.535e-4],
                  [2.500, 0.459e-4],
                  [2.750, 0.395e-4],
                  [3.000, 0.347e-4],
                  [3.250, 0.298e-4],
                  [3.500, 0.261e-4],
                  [3.750, 0.225e-4],
                  [4.000, 0.191e-4],
                  [4.250, 0.166e-4],
                  [4.500, 0.138e-4],
                  [4.750, 0.114e-4],
                  [5.000, 0.904e-5],
                  [5.250, 0.655e-5],
                  [5.500, 0.409e-5],
                  [5.750, 0.140e-5],
                  [6.000, 0.434e-7]])
    N = np.trapz(w[:, 1], x=w[:, 0])
    w[:, 1] /= N
    w_interp = interp1d(
        w[:, 0],
        w[:, 1],
        bounds_error=False,
        kind="cubic",
        fill_value=0
    )
    return w_interp(E)
