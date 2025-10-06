# kernel.py
#
# Energy deposition kernel data for varian 6MV source created with EDKnrc.
#
# Simulation details:
#
# Histories - 20000000
# Num cones - 48
# Angles - 3.75 degree spacing
# Num spheres - 24
# Medium - H20521ICRU
# ECUT - 0.521
# PCUT - 0.010
#
import numpy as np


class KernelMono:

    def __init__(self, egslst_path=None) -> None:
        if egslst_path is not None:
            self._from_egslst(egslst_path)
        else:
            raise NotImplementedError

    def _from_egslst(self, egslst_path: str):
        with open(egslst_path) as f:
            start = 0
            end = 0
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "KINETIC ENERGY OF THE INCIDENT BEAM:" in line:
                    self.energy = float(line.split()[-2])
                if "angle/deg.  radius/cm" in line:
                    start = i + 2
                if "END OF RUN" in line:
                    end = i - 1

            if start == 0 or end == 0:
                raise ImportError("Kernel info could not be found in .egslst file.")

            data_raw = np.array(
                [l.split()[:3] for l in lines[start:end]], dtype=np.float32
            )
            self.angles = np.unique(data_raw[:, 0])
            self.radii = np.unique(data_raw[:, 1])
            n_cones = len(self.angles)
            n_depths = len(self.radii)
            self.kernel = data_raw[:, 2].reshape((n_cones, n_depths), copy=True)

            r_edges = np.insert(self.radii, 0, 0)
            self.radii_centres = 0.5 * (r_edges[1:] + r_edges[:-1]) 
            angles_edges = np.insert(self.angles, 0, 0)
            self.angles_centres = 0.5 * (angles_edges[1:] + angles_edges[:-1]) 

            for i in range(self.kernel.shape[0]):
                w = 2 * np.pi * (np.cos(angles_edges[i] * np.pi / 180) - np.cos(angles_edges[i+1] * np.pi / 180))  # Angular weight for azimuthal symmetry
                self.kernel[i, :] = self.kernel[i, :] * w  # Convert to per unit solid angle
           
            kernel_interp = np.zeros((n_cones, 800))
            for i in range(self.kernel.shape[0]):
                kernel_interp[i, :] = np.interp(  # Resample to 0.25 mm
                    np.linspace(0.025, 20.0, 800),
                    self.radii,
                    self.kernel[i, :]
                )
            self.kernel = kernel_interp
            self.kernel = self.kernel / self.kernel.sum()

    def _spherical_voxel_volumes(self, r_edges, theta_edges):
        """
        Compute volumes of spherical coordinate bins with azimythal symmetry,

        Parameters
        ----------
        r_edges : array_like
            Radial bin edges (cm). Length N+1 for N bins.
        theta_edges : array_like
            Polar angle bin edges (radians, 0 = forward, pi = backward). Length M+1 for M bins.

        Returns
        -------
        volumes : ndarray
            Array of shape (M, N), giving volume (cm^3) of each voxel element.
        """
        # Radial shell volumes: (r_out^3 - r_in^3)/3
        dR = (r_edges[1:]**3 - r_edges[:-1]**3) / 3.0

        # Angular wedge factor: cos(theta_min) - cos(theta_max)
        dTheta = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])

        # Combine with 2pi for azimuthal symmetry
        return (2 * np.pi) * np.outer(dTheta, dR)
               
# class KernelPoly:

#     def __init__(self, angles, radii, kernel_diff) -> None:
#         self.angles = angles
#         self.radii = radii
#         self.kernel_diff = kernel_diff
#         self.kernel_diff = self.kernel_diff / self.kernel_diff.sum()  # normalise
#         self.kernel_cum = kernel_diff.cumsum(axis=1)
#         kernel_cum_interp = np.zeros((48, 5996))
#         for i in range(self.kernel_cum.shape[0]):
#             kernel_cum_interp[i, :] = np.interp(  # Resample to 0.1 mm
#                 np.linspace(0.05, 60.0, 5996),
#                 self.radii,
#                 self.kernel_cum[i, :]
#             )
#         self.kernel_cum = kernel_cum_interp
