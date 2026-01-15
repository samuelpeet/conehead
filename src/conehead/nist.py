#%%
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d


def mu_air(E, type="tot"):
    """Return mass attenuation coefficients for dry air near sea level based on NIST
    data.

    Parameters
    ----------
    E : ndarray
        Array of energies (units of MV)
    type : str, optional
        Type of mass attenuation coefficient to return. Options are 'tot' for
        total and 'en' for energy absorption coefficient. Default is 'tot'.

    Returns
    -------
    ndarray
        Array of interpolated mass attenuation coefficients
    """
    mu = np.array(
        [
            [1.00000E-03, 3.606E+03, 3.599E+03], 
            [1.50000E-03, 1.191E+03, 1.188E+03], 
            [2.00000E-03, 5.279E+02, 5.262E+02], 
            [3.00000E-03, 1.625E+02, 1.614E+02], 
            [3.20290E-03, 1.340E+02, 1.330E+02], 
            [3.20290E-03, 1.485E+02, 1.460E+02], 
            [4.00000E-03, 7.788E+01, 7.636E+01], 
            [5.00000E-03, 4.027E+01, 3.931E+01], 
            [6.00000E-03, 2.341E+01, 2.270E+01], 
            [8.00000E-03, 9.921E+00, 9.446E+00], 
            [1.00000E-02, 5.120E+00, 4.742E+00], 
            [1.50000E-02, 1.614E+00, 1.334E+00], 
            [2.00000E-02, 7.779E-01, 5.389E-01], 
            [3.00000E-02, 3.538E-01, 1.537E-01], 
            [4.00000E-02, 2.485E-01, 6.833E-02], 
            [5.00000E-02, 2.080E-01, 4.098E-02], 
            [6.00000E-02, 1.875E-01, 3.041E-02], 
            [8.00000E-02, 1.662E-01, 2.407E-02], 
            [1.00000E-01, 1.541E-01, 2.325E-02], 
            [1.50000E-01, 1.356E-01, 2.496E-02], 
            [2.00000E-01, 1.233E-01, 2.672E-02], 
            [3.00000E-01, 1.067E-01, 2.872E-02], 
            [4.00000E-01, 9.549E-02, 2.949E-02], 
            [5.00000E-01, 8.712E-02, 2.966E-02], 
            [6.00000E-01, 8.055E-02, 2.953E-02], 
            [8.00000E-01, 7.074E-02, 2.882E-02], 
            [1.00000E+00, 6.358E-02, 2.789E-02], 
            [1.25000E+00, 5.687E-02, 2.666E-02], 
            [1.50000E+00, 5.175E-02, 2.547E-02], 
            [2.00000E+00, 4.447E-02, 2.345E-02], 
            [3.00000E+00, 3.581E-02, 2.057E-02], 
            [4.00000E+00, 3.079E-02, 1.870E-02], 
            [5.00000E+00, 2.751E-02, 1.740E-02], 
            [6.00000E+00, 2.522E-02, 1.647E-02], 
            [8.00000E+00, 2.225E-02, 1.525E-02], 
            [1.00000E+01, 2.045E-02, 1.450E-02], 
            [1.50000E+01, 1.810E-02, 1.353E-02], 
            [2.00000E+01, 1.705E-02, 1.311E-02],            
        ],
        dtype=np.float32,
    )
    if type == "tot":
        return interp1d(mu[:, 0], mu[:, 1])(E)
    elif type == "en":
        return interp1d(mu[:, 0], mu[:, 2])(E)
    else:
        raise ValueError("Invalid type. Options are 'tot' or 'en'.")


def mu_lung(E, type="tot"):
    """Return mass attenuation coefficients for lung tissue based on NIST
    data.

    Parameters
    ----------
    E : ndarray
        Array of energies (units of MV)
    type : str, optional
        Type of mass attenuation coefficient to return. Options are 'tot' for
        total and 'en' for energy absorption coefficient. Default is 'tot'.

    Returns
    -------
    ndarray
        Array of interpolated mass attenuation coefficients
    """
    mu = np.array(
        [
            [1.00000E-03, 3.803E+03, 3.791E+03], 
            [1.03542E-03, 3.469E+03, 3.459E+03], 
            [1.07210E-03, 3.164E+03, 3.155E+03], 
            [1.07210E-03, 3.176E+03, 3.167E+03], 
            [1.50000E-03, 1.283E+03, 1.280E+03], 
            [2.00000E-03, 5.746E+02, 5.727E+02], 
            [2.14550E-03, 4.707E+02, 4.690E+02], 
            [2.14550E-03, 4.752E+02, 4.732E+02], 
            [2.30297E-03, 3.883E+02, 3.865E+02], 
            [2.47200E-03, 3.170E+02, 3.154E+02], 
            [2.47200E-03, 3.226E+02, 3.206E+02], 
            [2.64140E-03, 2.668E+02, 2.650E+02], 
            [2.82240E-03, 2.204E+02, 2.189E+02], 
            [2.82240E-03, 2.248E+02, 2.229E+02], 
            [3.00000E-03, 1.888E+02, 1.870E+02], 
            [3.60740E-03, 1.103E+02, 1.091E+02], 
            [3.60740E-03, 1.125E+02, 1.110E+02], 
            [4.00000E-03, 8.306E+01, 8.181E+01], 
            [5.00000E-03, 4.296E+01, 4.210E+01], 
            [6.00000E-03, 2.497E+01, 2.431E+01], 
            [8.00000E-03, 1.058E+01, 1.010E+01], 
            [1.00000E-02, 5.459E+00, 5.067E+00], 
            [1.50000E-02, 1.721E+00, 1.423E+00], 
            [2.00000E-02, 8.316E-01, 5.740E-01], 
            [3.00000E-02, 3.815E-01, 1.635E-01], 
            [4.00000E-02, 2.699E-01, 7.286E-02], 
            [5.00000E-02, 2.270E-01, 4.393E-02], 
            [6.00000E-02, 2.053E-01, 3.282E-02], 
            [8.00000E-02, 1.826E-01, 2.625E-02], 
            [1.00000E-01, 1.695E-01, 2.550E-02], 
            [1.50000E-01, 1.493E-01, 2.748E-02], 
            [2.00000E-01, 1.359E-01, 2.945E-02], 
            [3.00000E-01, 1.177E-01, 3.167E-02], 
            [4.00000E-01, 1.053E-01, 3.252E-02], 
            [5.00000E-01, 9.607E-02, 3.272E-02], 
            [6.00000E-01, 8.882E-02, 3.257E-02], 
            [8.00000E-01, 7.800E-02, 3.179E-02], 
            [1.00000E+00, 7.013E-02, 3.077E-02], 
            [1.25000E+00, 6.271E-02, 2.940E-02], 
            [1.50000E+00, 5.706E-02, 2.810E-02], 
            [2.00000E+00, 4.900E-02, 2.586E-02], 
            [3.00000E+00, 3.935E-02, 2.262E-02], 
            [4.00000E+00, 3.374E-02, 2.048E-02], 
            [5.00000E+00, 3.005E-02, 1.898E-02], 
            [6.00000E+00, 2.746E-02, 1.789E-02], 
            [8.00000E+00, 2.407E-02, 1.643E-02], 
            [1.00000E+01, 2.198E-02, 1.551E-02], 
            [1.50000E+01, 1.922E-02, 1.427E-02], 
            [2.00000E+01, 1.794E-02, 1.367E-02],             
        ],
        dtype=np.float32,
    )
    if type == "tot":
        return interp1d(mu[:, 0], mu[:, 1])(E)
    elif type == "en":
        return interp1d(mu[:, 0], mu[:, 2])(E)
    else:
        raise ValueError("Invalid type. Options are 'tot' or 'en'.")

    
def mu_adipose(E, type="tot"):
    """Return mass attenuation coefficients for adipose tissue based on NIST
    data.

    Parameters
    ----------
    E : ndarray
        Array of energies (units of MV)
    type : str, optional
        Type of mass attenuation coefficient to return. Options are 'tot' for
        total and 'en' for energy absorption coefficient. Default is 'tot'.

    Returns
    -------
    ndarray
        Array of interpolated mass attenuation coefficients
    """
    mu = np.array(
        [
            [1.00000E-03, 2.628E+03, 2.623E+03], 
            [1.03542E-03, 2.392E+03, 2.387E+03], 
            [1.07210E-03, 2.176E+03, 2.171E+03], 
            [1.07210E-03, 2.182E+03, 2.177E+03], 
            [1.50000E-03, 8.622E+02, 8.601E+02], 
            [2.00000E-03, 3.800E+02, 3.787E+02], 
            [2.47200E-03, 2.053E+02, 2.043E+02], 
            [2.47200E-03, 2.072E+02, 2.060E+02], 
            [2.64140E-03, 1.707E+02, 1.696E+02], 
            [2.82240E-03, 1.405E+02, 1.396E+02], 
            [2.82240E-03, 1.420E+02, 1.409E+02], 
            [3.00000E-03, 1.188E+02, 1.178E+02], 
            [4.00000E-03, 5.054E+01, 4.983E+01], 
            [5.00000E-03, 2.587E+01, 2.531E+01], 
            [6.00000E-03, 1.494E+01, 1.446E+01], 
            [8.00000E-03, 6.300E+00, 5.917E+00], 
            [1.00000E-02, 3.268E+00, 2.935E+00], 
            [1.50000E-02, 1.083E+00, 8.103E-01], 
            [2.00000E-02, 5.677E-01, 3.251E-01], 
            [3.00000E-02, 3.063E-01, 9.495E-02], 
            [4.00000E-02, 2.396E-01, 4.575E-02], 
            [5.00000E-02, 2.123E-01, 3.085E-02], 
            [6.00000E-02, 1.974E-01, 2.567E-02], 
            [8.00000E-02, 1.800E-01, 2.358E-02], 
            [1.00000E-01, 1.688E-01, 2.433E-02], 
            [1.50000E-01, 1.500E-01, 2.737E-02], 
            [2.00000E-01, 1.368E-01, 2.959E-02], 
            [3.00000E-01, 1.187E-01, 3.194E-02], 
            [4.00000E-01, 1.062E-01, 3.283E-02], 
            [5.00000E-01, 9.696E-02, 3.304E-02], 
            [6.00000E-01, 8.965E-02, 3.289E-02], 
            [8.00000E-01, 7.873E-02, 3.211E-02], 
            [1.00000E+00, 7.078E-02, 3.108E-02], 
            [1.25000E+00, 6.330E-02, 2.970E-02], 
            [1.50000E+00, 5.760E-02, 2.839E-02], 
            [2.00000E+00, 4.940E-02, 2.610E-02], 
            [3.00000E+00, 3.955E-02, 2.275E-02], 
            [4.00000E+00, 3.377E-02, 2.050E-02], 
            [5.00000E+00, 2.995E-02, 1.891E-02], 
            [6.00000E+00, 2.725E-02, 1.773E-02], 
            [8.00000E+00, 2.368E-02, 1.612E-02], 
            [1.00000E+01, 2.145E-02, 1.509E-02], 
            [1.50000E+01, 1.843E-02, 1.365E-02], 
            [2.00000E+01, 1.698E-02, 1.293E-02],
        ],
        dtype=np.float32,
    )
    if type == "tot":
        return interp1d(mu[:, 0], mu[:, 1])(E)
    elif type == "en":
        return interp1d(mu[:, 0], mu[:, 2])(E)
    else:
        raise ValueError("Invalid type. Options are 'tot' or 'en'.")

def mu_water(E, type="tot"):
    """Return mass attenuation coefficients for liquid water based on NIST
    data.

    Parameters
    ----------
    E : ndarray
        Array of energies (units of MV)
    type : str, optional
        Type of mass attenuation coefficient to return. Options are 'tot' for
        total and 'en' for energy absorption coefficient. Default is 'tot'.

    Returns
    -------
    ndarray
        Array of interpolated mass attenuation coefficients
    """
    mu = np.array(
        [
            [1.00000e-03, 4.078e03, 4.065e03],
            [1.50000e-03, 1.376e03, 1.372e03],
            [2.00000e-03, 6.173e02, 6.152e02],
            [3.00000e-03, 1.929e02, 1.917e02],
            [4.00000e-03, 8.278e01, 8.191e01],
            [5.00000e-03, 4.258e01, 4.188e01],
            [6.00000e-03, 2.464e01, 2.405e01],
            [8.00000e-03, 1.037e01, 9.915e00],
            [1.00000e-02, 5.329e00, 4.944e00],
            [1.50000e-02, 1.673e00, 1.374e00],
            [2.00000e-02, 8.096e-01, 5.503e-01],
            [3.00000e-02, 3.756e-01, 1.557e-01],
            [4.00000e-02, 2.683e-01, 6.947e-02],
            [5.00000e-02, 2.269e-01, 4.223e-02],
            [6.00000e-02, 2.059e-01, 3.190e-02],
            [8.00000e-02, 1.837e-01, 2.597e-02],
            [1.00000e-01, 1.707e-01, 2.546e-02],
            [1.50000e-01, 1.505e-01, 2.764e-02],
            [2.00000e-01, 1.370e-01, 2.967e-02],
            [3.00000e-01, 1.186e-01, 3.192e-02],
            [4.00000e-01, 1.061e-01, 3.279e-02],
            [5.00000e-01, 9.687e-02, 3.299e-02],
            [6.00000e-01, 8.956e-02, 3.284e-02],
            [8.00000e-01, 7.865e-02, 3.206e-02],
            [1.00000e00, 7.072e-02, 3.103e-02],
            [1.25000e00, 6.323e-02, 2.965e-02],
            [1.50000e00, 5.754e-02, 2.833e-02],
            [2.00000e00, 4.942e-02, 2.608e-02],
            [3.00000e00, 3.969e-02, 2.281e-02],
            [4.00000e00, 3.403e-02, 2.066e-02],
            [5.00000e00, 3.031e-02, 1.915e-02],
            [6.00000e00, 2.770e-02, 1.806e-02],
            [8.00000e00, 2.429e-02, 1.658e-02],
            [1.00000e01, 2.219e-02, 1.566e-02],
            [1.50000e01, 1.941e-02, 1.441e-02],
            [2.00000e01, 1.813e-02, 1.382e-02],
        ],
        dtype=np.float32,
    )
    if type == "tot":
        return interp1d(mu[:, 0], mu[:, 1])(E)
    elif type == "en":
        return interp1d(mu[:, 0], mu[:, 2])(E)
    else:
        raise ValueError("Invalid type. Options are 'tot' or 'en'.")


def mu_soft_tissue(E, type="tot"):
    """Return mass attenuation coefficients for soft tissue (ICRU Four-Component) based on NIST
    data.

    Parameters
    ----------
    E : ndarray
        Array of energies (units of MV)
    type : str, optional
        Type of mass attenuation coefficient to return. Options are 'tot' for
        total and 'en' for energy absorption coefficient. Default is 'tot'.

    Returns
    -------
    ndarray
        Array of interpolated mass attenuation coefficients
    """
    mu = np.array(
        [
            [1.00000E-03, 3.829E+03, 3.818E+03],
            [1.50000E-03, 1.286E+03, 1.283E+03],
            [2.00000E-03, 5.755E+02, 5.736E+02],
            [3.00000E-03, 1.792E+02, 1.781E+02],
            [4.00000E-03, 7.681E+01, 7.598E+01],
            [5.00000E-03, 3.947E+01, 3.880E+01],
            [6.00000E-03, 2.283E+01, 2.226E+01],
            [8.00000E-03, 9.604E+00, 9.163E+00],
            [1.00000E-02, 4.937E+00, 4.564E+00],
            [1.50000E-02, 1.558E+00, 1.266E+00],
            [2.00000E-02, 7.616E-01, 5.070E-01],
            [3.00000E-02, 3.604E-01, 1.438E-01],
            [4.00000E-02, 2.609E-01, 6.474E-02],
            [5.00000E-02, 2.223E-01, 3.987E-02],
            [6.00000E-02, 2.025E-01, 3.051E-02],
            [8.00000E-02, 1.813E-01, 2.530E-02],
            [1.00000E-01, 1.688E-01, 2.501E-02],
            [1.50000E-01, 1.490E-01, 2.732E-02],
            [2.00000E-01, 1.356E-01, 2.936E-02],
            [3.00000E-01, 1.175E-01, 3.161E-02],
            [4.00000E-01, 1.051E-01, 3.247E-02],
            [5.00000E-01, 9.593E-02, 3.267E-02],
            [6.00000E-01, 8.870E-02, 3.252E-02],
            [8.00000E-01, 7.789E-02, 3.175E-02],
            [1.00000E+00, 7.003E-02, 3.073E-02],
            [1.25000E+00, 6.262E-02, 2.937E-02],
            [1.50000E+00, 5.699E-02, 2.806E-02],
            [2.00000E+00, 4.893E-02, 2.582E-02],
            [3.00000E+00, 3.929E-02, 2.258E-02],
            [4.00000E+00, 3.367E-02, 2.044E-02],
            [5.00000E+00, 2.998E-02, 1.894E-02],
            [6.00000E+00, 2.739E-02, 1.785E-02],
            [8.00000E+00, 2.400E-02, 1.638E-02],
            [1.00000E+01, 2.191E-02, 1.546E-02],
            [1.50000E+01, 1.913E-02, 1.420E-02],
            [2.00000E+01, 1.785E-02, 1.360E-02],
        ],
        dtype=np.float32,
    )
    if type == "tot":
        return interp1d(mu[:, 0], mu[:, 1])(E)
    elif type == "en":
        return interp1d(mu[:, 0], mu[:, 2])(E)
    else:
        raise ValueError("Invalid type. Options are 'tot' or 'en'.")


def mu_muscle(E, type="tot"):
    """Return mass attenuation coefficients for skeletal muscle based on NIST
    data.

    Parameters
    ----------
    E : ndarray
        Array of energies (units of MV)
    type : str, optional
        Type of mass attenuation coefficient to return. Options are 'tot' for
        total and 'en' for energy absorption coefficient. Default is 'tot'.

    Returns
    -------
    ndarray
        Array of interpolated mass attenuation coefficients
    """
    mu = np.array(
        [
            [1.00000E-03, 3.719E+03, 3.709E+03], 
            [1.03542E-03, 3.393E+03, 3.383E+03], 
            [1.07210E-03, 3.094E+03, 3.085E+03], 
            [1.07210E-03, 3.100E+03, 3.091E+03], 
            [1.50000E-03, 1.251E+03, 1.247E+03], 
            [2.00000E-03, 5.594E+02, 5.574E+02], 
            [2.14550E-03, 4.581E+02, 4.564E+02], 
            [2.14550E-03, 4.626E+02, 4.606E+02], 
            [2.30297E-03, 3.776E+02, 3.762E+02], 
            [2.47200E-03, 3.085E+02, 3.069E+02], 
            [2.47200E-03, 3.140E+02, 3.121E+02], 
            [2.64140E-03, 2.597E+02, 2.579E+02], 
            [2.82240E-03, 2.145E+02, 2.130E+02], 
            [2.82240E-03, 2.160E+02, 2.143E+02], 
            [3.00000E-03, 1.812E+02, 1.796E+02], 
            [3.60740E-03, 1.057E+02, 1.046E+02], 
            [3.60740E-03, 1.100E+02, 1.083E+02], 
            [4.00000E-03, 8.127E+01, 7.992E+01], 
            [5.00000E-03, 4.206E+01, 4.116E+01], 
            [6.00000E-03, 2.446E+01, 2.377E+01], 
            [8.00000E-03, 1.037E+01, 9.888E+00], 
            [1.00000E-02, 5.356E+00, 4.964E+00], 
            [1.50000E-02, 1.693E+00, 1.396E+00], 
            [2.00000E-02, 8.205E-01, 5.638E-01], 
            [3.00000E-02, 3.783E-01, 1.610E-01], 
            [4.00000E-02, 2.685E-01, 7.192E-02], 
            [5.00000E-02, 2.262E-01, 4.349E-02], 
            [6.00000E-02, 2.048E-01, 3.258E-02], 
            [8.00000E-02, 1.823E-01, 2.615E-02], 
            [1.00000E-01, 1.693E-01, 2.544E-02], 
            [1.50000E-01, 1.492E-01, 2.745E-02], 
            [2.00000E-01, 1.358E-01, 2.942E-02], 
            [3.00000E-01, 1.176E-01, 3.164E-02], 
            [4.00000E-01, 1.052E-01, 3.249E-02], 
            [5.00000E-01, 9.598E-02, 3.269E-02], 
            [6.00000E-01, 8.874E-02, 3.254E-02], 
            [8.00000E-01, 7.793E-02, 3.177E-02], 
            [1.00000E+00, 7.007E-02, 3.074E-02], 
            [1.25000E+00, 6.265E-02, 2.938E-02], 
            [1.50000E+00, 5.701E-02, 2.808E-02], 
            [2.00000E+00, 4.896E-02, 2.584E-02], 
            [3.00000E+00, 3.931E-02, 2.259E-02], 
            [4.00000E+00, 3.369E-02, 2.045E-02], 
            [5.00000E+00, 3.000E-02, 1.895E-02], 
            [6.00000E+00, 2.741E-02, 1.786E-02], 
            [8.00000E+00, 2.401E-02, 1.639E-02], 
            [1.00000E+01, 2.192E-02, 1.547E-02], 
            [1.50000E+01, 1.915E-02, 1.421E-02], 
            [2.00000E+01, 1.786E-02, 1.361E-02], 
        ],
        dtype=np.float32,
    )
    if type == "tot":
        return interp1d(mu[:, 0], mu[:, 1])(E)
    elif type == "en":
        return interp1d(mu[:, 0], mu[:, 2])(E)
    else:
        raise ValueError("Invalid type. Options are 'tot' or 'en'.")


def mu_bone(E, type="tot"):
    """Return mass attenuation coefficients for bone based on NIST
    data.

    Parameters
    ----------
    E : ndarray
        Array of energies (units of MV)
    type : str, optional
        Type of mass attenuation coefficient to return. Options are 'tot' for
        total and 'en' for energy absorption coefficient. Default is 'tot'.

    Returns
    -------
    ndarray
        Array of interpolated mass attenuation coefficients
    """
    mu = np.array(
        [
            [1.00000E-03, 3.781E+03, 3.772E+03], 
            [1.03542E-03, 3.452E+03, 3.444E+03], 
            [1.07210E-03, 3.150E+03, 3.143E+03], 
            [1.07210E-03, 3.156E+03, 3.149E+03], 
            [1.18283E-03, 2.434E+03, 2.429E+03], 
            [1.30500E-03, 1.873E+03, 1.869E+03], 
            [1.30500E-03, 1.883E+03, 1.878E+03], 
            [1.50000E-03, 1.295E+03, 1.291E+03], 
            [2.00000E-03, 5.869E+02, 5.846E+02], 
            [2.14550E-03, 4.824E+02, 4.803E+02], 
            [2.14550E-03, 7.114E+02, 6.961E+02], 
            [2.30297E-03, 5.916E+02, 5.789E+02], 
            [2.47200E-03, 4.907E+02, 4.805E+02], 
            [2.47200E-03, 4.962E+02, 4.857E+02], 
            [3.00000E-03, 2.958E+02, 2.897E+02], 
            [4.00000E-03, 1.331E+02, 1.303E+02], 
            [4.03810E-03, 1.296E+02, 1.269E+02], 
            [4.03810E-03, 3.332E+02, 3.006E+02], 
            [5.00000E-03, 1.917E+02, 1.757E+02], 
            [6.00000E-03, 1.171E+02, 1.085E+02], 
            [8.00000E-03, 5.323E+01, 4.987E+01], 
            [1.00000E-02, 2.851E+01, 2.680E+01], 
            [1.50000E-02, 9.032E+00, 8.388E+00], 
            [2.00000E-02, 4.001E+00, 3.601E+00], 
            [3.00000E-02, 1.331E+00, 1.070E+00], 
            [4.00000E-02, 6.655E-01, 4.507E-01], 
            [5.00000E-02, 4.242E-01, 2.336E-01], 
            [6.00000E-02, 3.148E-01, 1.400E-01], 
            [8.00000E-02, 2.229E-01, 6.896E-02], 
            [1.00000E-01, 1.855E-01, 4.585E-02], 
            [1.50000E-01, 1.480E-01, 3.183E-02], 
            [2.00000E-01, 1.309E-01, 3.003E-02], 
            [3.00000E-01, 1.113E-01, 3.032E-02], 
            [4.00000E-01, 9.908E-02, 3.069E-02], 
            [5.00000E-01, 9.022E-02, 3.073E-02], 
            [6.00000E-01, 8.332E-02, 3.052E-02], 
            [8.00000E-01, 7.308E-02, 2.973E-02], 
            [1.00000E+00, 6.566E-02, 2.875E-02], 
            [1.25000E+00, 5.871E-02, 2.745E-02], 
            [1.50000E+00, 5.346E-02, 2.623E-02], 
            [2.00000E+00, 4.607E-02, 2.421E-02], 
            [3.00000E+00, 3.745E-02, 2.145E-02], 
            [4.00000E+00, 3.257E-02, 1.975E-02], 
            [5.00000E+00, 2.946E-02, 1.864E-02], 
            [6.00000E+00, 2.734E-02, 1.788E-02], 
            [8.00000E+00, 2.467E-02, 1.695E-02], 
            [1.00000E+01, 2.314E-02, 1.644E-02], 
            [1.50000E+01, 2.132E-02, 1.587E-02], 
            [2.00000E+01, 2.068E-02, 1.568E-02],
        ],
        dtype=np.float32,
    )
    if type == "tot":
        return interp1d(mu[:, 0], mu[:, 1])(E)
    elif type == "en":
        return interp1d(mu[:, 0], mu[:, 2])(E)
    else:
        raise ValueError("Invalid type. Options are 'tot' or 'en'.")


def mu_Al(E, type="tot"):
    """Return mass attenuation coefficients for aluminium based on NIST data.

    Parameters
    ----------
    E : ndarray
        Array of energies (units of MV)
    type : str, optional
        Type of mass attenuation coefficient to return. Options are 'tot' for
        total and 'en' for energy absorption coefficient. Default is 'tot'.

    Returns
    -------
    ndarray
        Array of interpolated mass attenuation coefficients
    """
    mu = np.array(
        [
            [1.00000e-03, 1.185e03, 1.183e03],
            [1.50000e-03, 4.022e02, 4.001e02],
            [1.55960e-03, 3.621e02, 3.600e02],
            [1.55960e-03, 3.957e03, 3.829e03],
            [2.00000e-03, 2.263e03, 2.204e03],
            [3.00000e-03, 7.880e02, 7.732e02],
            [4.00000e-03, 3.605e02, 3.545e02],
            [5.00000e-03, 1.934e02, 1.902e02],
            [6.00000e-03, 1.153e02, 1.133e02],
            [8.00000e-03, 5.033e01, 4.918e01],
            [1.00000e-02, 2.623e01, 2.543e01],
            [1.50000e-02, 7.955e00, 7.487e00],
            [2.00000e-02, 3.441e00, 3.094e00],
            [3.00000e-02, 1.128e00, 8.778e-01],
            [4.00000e-02, 5.685e-01, 3.601e-01],
            [5.00000e-02, 3.681e-01, 1.840e-01],
            [6.00000e-02, 2.778e-01, 1.099e-01],
            [8.00000e-02, 2.018e-01, 5.511e-02],
            [1.00000e-01, 1.704e-01, 3.794e-02],
            [1.50000e-01, 1.378e-01, 2.827e-02],
            [2.00000e-01, 1.223e-01, 2.745e-02],
            [3.00000e-01, 1.042e-01, 2.816e-02],
            [4.00000e-01, 9.276e-02, 2.862e-02],
            [5.00000e-01, 8.445e-02, 2.868e-02],
            [6.00000e-01, 7.802e-02, 2.851e-02],
            [8.00000e-01, 6.841e-02, 2.778e-02],
            [1.00000e00, 6.146e-02, 2.686e-02],
            [1.25000e00, 5.496e-02, 2.565e-02],
            [1.50000e00, 5.006e-02, 2.451e-02],
            [2.00000e00, 4.324e-02, 2.266e-02],
            [3.00000e00, 3.541e-02, 2.024e-02],
            [4.00000e00, 3.106e-02, 1.882e-02],
            [5.00000e00, 2.836e-02, 1.795e-02],
            [6.00000e00, 2.655e-02, 1.739e-02],
            [8.00000e00, 2.437e-02, 1.678e-02],
            [1.00000e01, 2.318e-02, 1.650e-02],
            [1.50000e01, 2.195e-02, 1.631e-02],
            [2.00000e01, 2.168e-02, 1.633e-02],
        ],
        dtype=np.float32,
    )
    if type == "tot":
        return interp1d(mu[:, 0], mu[:, 1])(E)
    elif type == "en":
        return interp1d(mu[:, 0], mu[:, 2])(E)
    else:
        raise ValueError("Invalid type. Options are 'tot' or 'en'.")


def mu_Fe(E, type="tot"):
    """Return mass attenuation coefficients for iron based on NIST data.

    Parameters
    ----------
    E : ndarray
        Array of energies (units of MV)
    type : str, optional
        Type of mass attenuation coefficient to return. Options are 'tot' for
        total and 'en' for energy absorption coefficient. Default is 'tot'.

    Returns
    -------
    ndarray
        Array of interpolated mass attenuation coefficients
    """
    mu = np.array(
        [
            [1.00000E-03, 9.085E+03, 9.052E+03], 
            [1.50000E-03, 3.399E+03, 3.388E+03], 
            [2.00000E-03, 1.626E+03, 1.620E+03], 
            [3.00000E-03, 5.576E+02, 5.535E+02], 
            [4.00000E-03, 2.567E+02, 2.536E+02], 
            [5.00000E-03, 1.398E+02, 1.372E+02], 
            [6.00000E-03, 8.484E+01, 8.265E+01], 
            [7.11200E-03, 5.319E+01, 5.133E+01], 
            [7.11200E-03, 4.076E+02, 2.978E+02], 
            [8.00000E-03, 3.056E+02, 2.316E+02], 
            [1.00000E-02, 1.706E+02, 1.369E+02], 
            [1.50000E-02, 5.708E+01, 4.896E+01], 
            [2.00000E-02, 2.568E+01, 2.260E+01], 
            [3.00000E-02, 8.176E+00, 7.251E+00], 
            [4.00000E-02, 3.629E+00, 3.155E+00], 
            [5.00000E-02, 1.958E+00, 1.638E+00], 
            [6.00000E-02, 1.205E+00, 9.555E-01], 
            [8.00000E-02, 5.952E-01, 4.104E-01], 
            [1.00000E-01, 3.717E-01, 2.177E-01], 
            [1.50000E-01, 1.964E-01, 7.961E-02], 
            [2.00000E-01, 1.460E-01, 4.825E-02], 
            [3.00000E-01, 1.099E-01, 3.361E-02], 
            [4.00000E-01, 9.400E-02, 3.039E-02], 
            [5.00000E-01, 8.414E-02, 2.914E-02], 
            [6.00000E-01, 7.704E-02, 2.836E-02], 
            [8.00000E-01, 6.699E-02, 2.714E-02], 
            [1.00000E+00, 5.995E-02, 2.603E-02], 
            [1.25000E+00, 5.350E-02, 2.472E-02], 
            [1.50000E+00, 4.883E-02, 2.360E-02], 
            [2.00000E+00, 4.265E-02, 2.199E-02], 
            [3.00000E+00, 3.621E-02, 2.042E-02], 
            [4.00000E+00, 3.312E-02, 1.990E-02], 
            [5.00000E+00, 3.146E-02, 1.983E-02], 
            [6.00000E+00, 3.057E-02, 1.997E-02], 
            [8.00000E+00, 2.991E-02, 2.050E-02], 
            [1.00000E+01, 2.994E-02, 2.108E-02], 
            [1.50000E+01, 3.092E-02, 2.221E-02], 
            [2.00000E+01, 3.224E-02, 2.292E-02],
        ],
        dtype=np.float32,
    )
    if type == "tot":
        return interp1d(mu[:, 0], mu[:, 1])(E)
    elif type == "en":
        return interp1d(mu[:, 0], mu[:, 2])(E)
    else:
        raise ValueError("Invalid type. Options are 'tot' or 'en'.")


def mu_Au(E, type="tot"):
    """Return mass attenuation coefficients for gold based on NIST data.

    Parameters
    ----------
    E : ndarray
        Array of energies (units of MV)
    type : str, optional
        Type of mass attenuation coefficient to return. Options are 'tot' for
        total and 'en' for energy absorption coefficient. Default is 'tot'.

    Returns
    -------
    ndarray
        Array of interpolated mass attenuation coefficients
    """
    mu = np.array(
        [
            [1.00000E-03, 4.652E+03, 4.639E+03], 
            [1.50000E-03, 2.089E+03, 2.076E+03], 
            [2.00000E-03, 1.137E+03, 1.125E+03], 
            [2.20570E-03, 9.187E+02, 9.074E+02], 
            [2.20570E-03, 9.971E+02, 9.836E+02], 
            [2.24799E-03, 1.386E+03, 1.360E+03], 
            [2.29110E-03, 2.258E+03, 2.208E+03], 
            [2.29110E-03, 2.389E+03, 2.336E+03], 
            [2.50689E-03, 2.380E+03, 2.325E+03], 
            [2.74300E-03, 2.203E+03, 2.154E+03], 
            [2.74300E-03, 2.541E+03, 2.484E+03], 
            [3.00000E-03, 2.049E+03, 2.005E+03], 
            [3.14780E-03, 1.822E+03, 1.783E+03], 
            [3.14780E-03, 1.933E+03, 1.892E+03], 
            [3.28343E-03, 1.748E+03, 1.710E+03], 
            [3.42490E-03, 1.585E+03, 1.552E+03], 
            [3.42490E-03, 1.652E+03, 1.618E+03], 
            [4.00000E-03, 1.144E+03, 1.120E+03], 
            [5.00000E-03, 6.661E+02, 6.512E+02], 
            [6.00000E-03, 4.253E+02, 4.143E+02], 
            [8.00000E-03, 2.072E+02, 1.999E+02], 
            [1.00000E-02, 1.181E+02, 1.126E+02], 
            [1.19187E-02, 7.582E+01, 7.129E+01], 
            [1.19187E-02, 1.870E+02, 1.521E+02], 
            [1.27940E-02, 1.546E+02, 1.272E+02], 
            [1.37336E-02, 1.283E+02, 1.066E+02], 
            [1.37336E-02, 1.764E+02, 1.379E+02], 
            [1.40398E-02, 1.766E+02, 1.317E+02], 
            [1.43528E-02, 1.588E+02, 1.252E+02], 
            [1.43528E-02, 1.830E+02, 1.432E+02], 
            [1.50000E-02, 1.637E+02, 1.294E+02], 
            [2.00000E-02, 7.883E+01, 6.522E+01], 
            [3.00000E-02, 2.752E+01, 2.349E+01], 
            [4.00000E-02, 1.298E+01, 1.109E+01], 
            [5.00000E-02, 7.256E+00, 6.124E+00], 
            [6.00000E-02, 4.528E+00, 3.751E+00], 
            [8.00000E-02, 2.185E+00, 1.720E+00], 
            [8.07249E-02, 2.137E+00, 1.678E+00], 
            [8.07249E-02, 8.904E+00, 2.512E+00], 
            [1.00000E-01, 5.158E+00, 2.074E+00], 
            [1.50000E-01, 1.860E+00, 1.026E+00], 
            [2.00000E-01, 9.214E-01, 5.563E-01], 
            [3.00000E-01, 3.744E-01, 2.289E-01], 
            [4.00000E-01, 2.180E-01, 1.274E-01], 
            [5.00000E-01, 1.530E-01, 8.523E-02], 
            [6.00000E-01, 1.194E-01, 6.409E-02], 
            [8.00000E-01, 8.603E-02, 4.427E-02], 
            [1.00000E+00, 6.953E-02, 3.525E-02], 
            [1.25000E+00, 5.794E-02, 2.915E-02], 
            [1.50000E+00, 5.167E-02, 2.593E-02], 
            [2.00000E+00, 4.570E-02, 2.333E-02], 
            [3.00000E+00, 4.201E-02, 2.302E-02], 
            [4.00000E+00, 4.166E-02, 2.432E-02], 
            [5.00000E+00, 4.239E-02, 2.582E-02], 
            [6.00000E+00, 4.355E-02, 2.725E-02], 
            [8.00000E+00, 4.633E-02, 2.968E-02], 
            [1.00000E+01, 4.926E-02, 3.159E-02], 
            [1.50000E+01, 5.598E-02, 3.450E-02], 
            [2.00000E+01, 6.136E-02, 3.565E-02],
        ],
        dtype=np.float32,
    )
    if type == "tot":
        return interp1d(mu[:, 0], mu[:, 1])(E)
    elif type == "en":
        return interp1d(mu[:, 0], mu[:, 2])(E)
    else:
        raise ValueError("Invalid type. Options are 'tot' or 'en'.")


def mu_Os(E, type="tot"):
    """Return mass attenuation coefficients for osmium based on NIST data.

    Parameters
    ----------
    E : ndarray
        Array of energies (units of MV)
    type : str, optional
        Type of mass attenuation coefficient to return. Options are 'tot' for
        total and 'en' for energy absorption coefficient. Default is 'tot'.

    Returns
    -------
    ndarray
        Array of interpolated mass attenuation coefficients
    """
    mu = np.array(
        [
            [1.00000E-03, 4.032E+03, 4.019E+03], 
            [1.50000E-03, 1.801E+03, 1.790E+03], 
            [1.96010E-03, 1.023E+03, 1.012E+03], 
            [1.96010E-03, 2.003E+03, 1.969E+03], 
            [2.00000E-03, 2.218E+03, 2.179E+03], 
            [2.03080E-03, 2.622E+03, 2.573E+03], 
            [2.03080E-03, 2.864E+03, 2.810E+03], 
            [2.23385E-03, 2.852E+03, 2.797E+03], 
            [2.45720E-03, 2.546E+03, 2.498E+03], 
            [2.45720E-03, 2.948E+03, 2.893E+03], 
            [2.61935E-03, 2.524E+03, 2.478E+03], 
            [2.79220E-03, 2.161E+03, 2.123E+03], 
            [2.79220E-03, 2.296E+03, 2.255E+03], 
            [3.00000E-03, 1.938E+03, 1.904E+03], 
            [3.04850E-03, 1.869E+03, 1.836E+03], 
            [3.04850E-03, 1.949E+03, 1.915E+03], 
            [4.00000E-03, 1.023E+03, 1.004E+03], 
            [5.00000E-03, 5.936E+02, 5.813E+02], 
            [6.00000E-03, 3.776E+02, 3.682E+02], 
            [8.00000E-03, 1.836E+02, 1.770E+02], 
            [1.00000E-02, 1.045E+02, 9.940E+01], 
            [1.08709E-02, 8.458E+01, 7.994E+01], 
            [1.08709E-02, 2.121E+02, 1.762E+02], 
            [1.16033E-02, 1.860E+02, 1.495E+02], 
            [1.23850E-02, 1.503E+02, 1.271E+02], 
            [1.23850E-02, 2.060E+02, 1.654E+02], 
            [1.26731E-02, 1.956E+02, 1.576E+02], 
            [1.29680E-02, 1.846E+02, 1.494E+02], 
            [1.29680E-02, 2.129E+02, 1.711E+02], 
            [1.50000E-02, 1.478E+02, 1.218E+02], 
            [2.00000E-02, 7.039E+01, 5.999E+01], 
            [3.00000E-02, 2.443E+01, 2.120E+01], 
            [4.00000E-02, 1.149E+01, 9.907E+00], 
            [5.00000E-02, 6.414E+00, 5.437E+00], 
            [6.00000E-02, 4.002E+00, 3.314E+00], 
            [7.38708E-02, 2.360E+00, 1.879E+00], 
            [7.38708E-02, 1.016E+01, 2.888E+00], 
            [8.00000E-02, 8.290E+00, 2.756E+00], 
            [1.00000E-01, 4.696E+00, 2.092E+00], 
            [1.50000E-01, 1.680E+00, 9.696E-01], 
            [2.00000E-01, 8.327E-01, 5.150E-01], 
            [3.00000E-01, 3.414E-01, 2.085E-01], 
            [4.00000E-01, 2.011E-01, 1.161E-01], 
            [5.00000E-01, 1.428E-01, 7.813E-02], 
            [6.00000E-01, 1.125E-01, 5.923E-02], 
            [8.00000E-01, 8.224E-02, 4.157E-02], 
            [1.00000E+00, 6.705E-02, 3.351E-02], 
            [1.25000E+00, 5.625E-02, 2.802E-02], 
            [1.50000E+00, 5.034E-02, 2.511E-02], 
            [2.00000E+00, 4.458E-02, 2.272E-02], 
            [3.00000E+00, 4.100E-02, 2.248E-02], 
            [4.00000E+00, 4.065E-02, 2.376E-02], 
            [5.00000E+00, 4.134E-02, 2.523E-02], 
            [6.00000E+00, 4.244E-02, 2.663E-02], 
            [8.00000E+00, 4.511E-02, 2.901E-02], 
            [1.00000E+01, 4.791E-02, 3.086E-02], 
            [1.50000E+01, 5.442E-02, 3.375E-02], 
            [2.00000E+01, 5.956E-02, 3.486E-02],
        ],
        dtype=np.float32,
    )
    if type == "tot":
        return interp1d(mu[:, 0], mu[:, 1])(E)
    elif type == "en":
        return interp1d(mu[:, 0], mu[:, 2])(E)
    else:
        raise ValueError("Invalid type. Options are 'tot' or 'en'.")
    

def map_materials(densities: npt.NDArray[np.float32]) -> npt.NDArray[np.int8]:
    """Map densities to material indices.

    Parameters
    ----------
    densities : ndarray
        Array of densities (g/cm^3)

    Returns
    -------
    ndarray
        Array of material indices
    """
    # Define material boundaries
    air_lung_bound = 0.13  # g/cm^3
    lung_adipose_bound = 0.605  # g/cm^3
    adipose_soft_tissue_bound = 0.975  # g/cm^3
    soft_tissue_muscle_bound = 1.025  # g/cm^3
    muscle_bone_bound = 1.185  # g/cm^3
    bone_Al_bound = 2.2  # g/cm^3
    Al_Fe_bound = 5.0  # g/cm^3
    Fe_Au_bound = 12.0  # g/cm^3
    Au_Os_bound = 21.0  # g/cm^3


    materials = np.zeros_like(densities, dtype=np.int8)
    materials[(densities < air_lung_bound)] = 0  # Air
    materials[(densities >= air_lung_bound) & (densities < lung_adipose_bound)] = 1  # Lung
    materials[(densities >= lung_adipose_bound) & (densities < adipose_soft_tissue_bound)] = 2  # Adipose
    materials[(densities >= adipose_soft_tissue_bound) & (densities < soft_tissue_muscle_bound)] = 3  # Soft Tissue
    materials[(densities >= soft_tissue_muscle_bound) & (densities < muscle_bone_bound)] = 4  # Muscle
    materials[(densities >= muscle_bone_bound) & (densities < bone_Al_bound)] = 5  # Bone
    materials[(densities >= bone_Al_bound) & (densities < Al_Fe_bound)] = 6  # Aluminium
    materials[(densities >= Al_Fe_bound) & (densities < Fe_Au_bound)] = 7  # Iron
    materials[(densities >= Fe_Au_bound) & (densities < Au_Os_bound)] = 8  # Gold
    materials[(densities >= Au_Os_bound)] = 9  # Osmium
    return materials


# Implement a function that constructs and returns a 2D table. The 10 rows of the table represent the energy spectrum used in this program (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 MeV). The 10 columns of table represent the mu_en/rho values of the 10 materials above (air, lung, adipose, soft tissue, muscle, bone, aluminium, iron, gold, osmium) at those energies. The function should return the table as a numpy array.
def mass_energy_absorption_coefficient_table() -> npt.NDArray[np.float32]:
    """Return a 2D table of mass energy absorption coefficients (mu_en/rho).
    
    Rows represent energies: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 MeV
    Columns represent materials: air, lung, adipose, soft tissue, muscle, bone, Al, Fe, Au, Os
    
    Returns
    -------
    ndarray
        Shape (12, 10) table of mu_en/rho values in cm^2/g
    """
    energies = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0], dtype=np.float32)
    
    table = np.zeros((len(energies), 10), dtype=np.float32)
    
    # Column 0: Air
    table[:, 0] = mu_air(energies, type="en")
    # Column 1: Lung
    table[:, 1] = mu_lung(energies, type="en")
    # Column 2: Adipose
    table[:, 2] = mu_adipose(energies, type="en")
    # Column 3: Soft tissue
    table[:, 3] = mu_soft_tissue(energies, type="en")
    # Column 4: Muscle
    table[:, 4] = mu_muscle(energies, type="en")
    # Column 5: Bone
    table[:, 5] = mu_bone(energies, type="en")
    # Column 6: Aluminium
    table[:, 6] = mu_Al(energies, type="en")
    # Column 7: Iron
    table[:, 7] = mu_Fe(energies, type="en")
    # Column 8: Gold
    table[:, 8] = mu_Au(energies, type="en")
    # Column 9: Osmium
    table[:, 9] = mu_Os(energies, type="en")
    
    return table


def mass_energy_coefficient_table() -> npt.NDArray[np.float32]:
    """Return a 2D table of mass attenuation coefficients (mu/rho).
    
    Rows represent energies: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 MeV
    Columns represent materials: air, lung, adipose, soft tissue, muscle, bone, Al, Fe, Au, Os
    
    Returns
    -------
    ndarray
        Shape (12, 10) table of mu/rho values in cm^2/g
    """
    energies = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0], dtype=np.float32)
    
    table = np.zeros((len(energies), 10), dtype=np.float32)
    
    # Column 0: Air
    table[:, 0] = mu_air(energies, type="tot")
    # Column 1: Lung
    table[:, 1] = mu_lung(energies, type="tot")
    # Column 2: Adipose
    table[:, 2] = mu_adipose(energies, type="tot")
    # Column 3: Soft tissue
    table[:, 3] = mu_soft_tissue(energies, type="tot")
    # Column 4: Muscle
    table[:, 4] = mu_muscle(energies, type="tot")
    # Column 5: Bone
    table[:, 5] = mu_bone(energies, type="tot")
    # Column 6: Aluminium
    table[:, 6] = mu_Al(energies, type="tot")
    # Column 7: Iron
    table[:, 7] = mu_Fe(energies, type="tot")
    # Column 8: Gold
    table[:, 8] = mu_Au(energies, type="tot")
    # Column 9: Osmium
    table[:, 9] = mu_Os(energies, type="tot")
    
    return table
# %%
