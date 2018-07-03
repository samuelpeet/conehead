# Spectral information for a 6MV Varian Clinac.
#
# From article:
# E S M Ali and D W O Rogers 2012 Phys. Med. Biol. 57 31
import numpy as np
from conehead.nist import mu_W, mu_Al


def psi_E(E):
    # Equation 12 of Table 2.
    C_1 = 1.222
    C_2 = 5.147
    C_3 = -1.186
    E_e = 5.76
    psi_thin = (1 + C_3 * E / E_e + np.power(E / E_e, 2)) * (np.log(E_e * (E_e - E) / E + 1.65) - 0.5)
    result = psi_thin * np.exp(-mu_W(E) * np.power(C_1, 2) - mu_Al(E) * np.power(C_2, 2))
    result[result < 0] = 0
    return result
