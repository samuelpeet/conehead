"""
Kernel module for radiotherapy dose calculation.

This module provides classes for loading and processing mono-energetic and
poly-energetic collapsed-cone convolution kernels used in dose calculations.
Kernels are typically generated from Monte Carlo simulations (e.g., EGSnrc)
and describe energy deposition as a function of angle and radiological depth.
"""

import numpy as np
from importlib.resources import files
from conehead.nist import mu_water


class KernelMono:
    """
    Mono-energetic collapsed-cone convolution kernel.

    This class loads and processes a single-energy kernel from an EGSnrc
    .egslst output file. The kernel describes energy deposition as a function
    of polar angle (cone opening) and radial distance from the interaction site.

    Attributes
    ----------
    energy : float
        Kinetic energy of the incident beam in MeV.
    data_raw : ndarray
        Raw kernel data (angle, radius, deposition) from the .egslst file.
    angles : ndarray
        Unique polar angles (degrees) at which the kernel is sampled.
    radii : ndarray
        Unique radii (cm) at which the kernel is sampled.
    kernel : ndarray, shape (n_cones, n_depths)
        Normalized kernel values. Entry [i, j] is the dose kernel at cone i
        and depth j, normalized such that the kernel integrates to 1.
    radii_centres : ndarray
        Radial bin centers (cm).
    angles_edges : ndarray
        Polar angle bin edges (degrees).
    angles_centres : ndarray
        Polar angle bin centers (degrees).
    omegas : ndarray
        Solid angle weights for each cone (steradian), computed from angle edges.

    Parameters
    ----------
    egslst_path : str, optional
        Path to the EGSnrc .egslst file. If None, raises NotImplementedError.

    Raises
    ------
    NotImplementedError
        If egslst_path is None (other initialization methods not yet supported).
    ImportError
        If the kernel data cannot be found in the .egslst file.
    """

    def __init__(self, egslst_path=None) -> None:
        if egslst_path is not None:
            self._from_egslst(egslst_path)
        else:
            raise NotImplementedError

    def _from_egslst(self, egslst_path: str):
        """
        Load and process kernel data from an EGSnrc .egslst file.

        This method parses the .egslst output to extract the incident beam
        energy and the angle/radius/deposition data. It then computes bin
        edges and centers, solid angle weights, normalizes the kernel by
        shell volume, and ensures the kernel integrates to unity.

        Parameters
        ----------
        egslst_path : str
            Path to the .egslst file.

        Raises
        ------
        ImportError
            If "KINETIC ENERGY OF THE INCIDENT BEAM:" or the data table
            cannot be found in the file.
        """
        with open(egslst_path) as f:
            start = 0
            end = 0
            lines = f.readlines()

            # Parse the file to find the beam energy and data table bounds
            for i, line in enumerate(lines):
                if "KINETIC ENERGY OF THE INCIDENT BEAM:" in line:
                    self.energy = float(line.split()[-2])
                if "angle/deg.  radius/cm" in line:
                    start = i + 2  # Data starts 2 lines after header
                if "END OF RUN" in line:
                    end = i - 1  # Data ends 1 line before END OF RUN

            if start == 0 or end == 0:
                raise ImportError("Kernel info could not be found in .egslst file.")

            # Parse data table: angle, radius, deposition (first 3 columns)
            self.data_raw = np.array(
                [line.split()[:3] for line in lines[start:end]], dtype=np.float32
            )
            self.angles = np.unique(self.data_raw[:, 0])
            self.radii = np.unique(self.data_raw[:, 1])
            n_cones = len(self.angles)
            n_depths = len(self.radii)
            self.kernel = self.data_raw[:, 2].reshape((n_cones, n_depths), copy=True)

            # Compute bin edges and centers for radius and angle
            r_edges = np.insert(self.radii, 0, 0)
            self.radii_centres = 0.5 * (r_edges[1:] + r_edges[:-1])
            self.angles_edges = np.insert(self.angles, 0, 0)
            self.angles_centres = 0.5 * (self.angles_edges[1:] + self.angles_edges[:-1])

            # Solid angle weights: Ω = 2π (cos(θ₁) - cos(θ₂))
            self.omegas = (
                2
                * np.pi
                * (
                    np.cos(np.radians(self.angles_edges[:-1]))
                    - np.cos(np.radians(self.angles_edges[1:]))
                )
            )

            # Normalize kernel by shell volume: dV = (r₂³ - r₁³)/3 · dΩ
            for i in range(self.kernel.shape[0]):
                dR = (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3
                dPhi = self.omegas
                self.kernel[i, :] = self.kernel[i, :] / (dR * dPhi[i])

            # Normalize so kernel integrates to 1
            self.kernel = self.kernel / self.kernel.sum()

        # Compute cumulative kernel along depth axis
        # kernel_cum[i, j] = integral of differential kernel from depth 0 to j
        self.kernel_cum = np.cumsum(self.kernel, axis=1)


class Kernel:
    """
    Poly-energetic collapsed-cone convolution kernel.

    This class combines multiple mono-energetic kernels into a single
    poly-energetic kernel weighted by an energy spectrum. The resulting
    kernel is used in collapsed-cone dose convolution algorithms.

    Attributes
    ----------
    phis : ndarray
        Polar angle bin centers (degrees) inherited from mono kernels.
    thetas : ndarray
        Azimuthal angle samples (degrees). Default is 16 evenly-spaced
        angles from 0 to 360 (exclusive of 360).
    values : ndarray, shape (n_phis, n_depths)
        Combined poly-energetic kernel values at the surface (depth 0),
        normalized to integrate to 1 and divided by theta sampling.
    omegas : ndarray
        Solid angle weights per phi bin, adjusted for theta sampling.
    values_depth : ndarray, shape (n_depth_bins_depth, n_phis, n_depths)
        Depth-dependent cumulative poly-energetic kernel table. First axis corresponds
        to water-equivalent depth from source to interaction site. Cumulative kernels
        should be sampled via difference: delta_dose = kernel[depth_i] - kernel[depth_i-1].
    values_depth_diff : ndarray, shape (n_depth_bins_depth, n_phis, n_depths)
        Depth-dependent differential (original) poly-energetic kernel table.
    spectrum_depth_res_cm : float
        Resolution of the depth-dependent spectrum bins (cm).
    max_spectrum_depth_cm : float
        Maximum water-equivalent depth captured in the spectrum bins (cm).
    n_spectrum_depth_bins : int
        Number of depth bins along the spectrum-hardening axis.

    Parameters
    ----------
    settings : dict
        Dictionary containing the energy spectrum. Expected keys:
        - 'energy_spectrum' : dict
            - 'energies' : list of float
                Energy bin values in MeV (e.g., [0.5, 1.0, ..., 6.0]).
            - 'weights' : list of float
                Corresponding normalized weights for each energy bin.

    Notes
    -----
    The class loads 12 mono-energetic kernels (0.5 MeV to 6.0 MeV in 0.5 MeV
    steps) and combines them using the provided energy spectrum. The
    azimuthal sampling uses 16 theta angles uniformly distributed around the
    cone axis; the kernel value is divided by the number of theta samples to
    ensure correct integration over the full solid angle.
    """

    def __init__(self, field_size: float, settings: dict) -> None:
        # Load mono-energetic kernels for 0.5 MeV to 6.0 MeV (12 kernels)
        kernels = [
            KernelMono(files("conehead.kernels").joinpath("0.5MeV/0.5MeV.egslst")),
            KernelMono(files("conehead.kernels").joinpath("1.0MeV/1.0MeV.egslst")),
            KernelMono(files("conehead.kernels").joinpath("1.5MeV/1.5MeV.egslst")),
            KernelMono(files("conehead.kernels").joinpath("2.0MeV/2.0MeV.egslst")),
            KernelMono(files("conehead.kernels").joinpath("2.5MeV/2.5MeV.egslst")),
            KernelMono(files("conehead.kernels").joinpath("3.0MeV/3.0MeV.egslst")),
            KernelMono(files("conehead.kernels").joinpath("3.5MeV/3.5MeV.egslst")),
            KernelMono(files("conehead.kernels").joinpath("4.0MeV/4.0MeV.egslst")),
            KernelMono(files("conehead.kernels").joinpath("4.5MeV/4.5MeV.egslst")),
            KernelMono(files("conehead.kernels").joinpath("5.0MeV/5.0MeV.egslst")),
            KernelMono(files("conehead.kernels").joinpath("5.5MeV/5.5MeV.egslst")),
            KernelMono(files("conehead.kernels").joinpath("6.0MeV/6.0MeV.egslst")),
        ]

        # Interpolate energy spectrum weights based on field size
        energies = np.asarray(settings["energy_spectrum"]["energies"], dtype=np.float32)
        weights_3 = np.asarray(settings["energy_spectrum"]["weights_3"], dtype=np.float32)
        weights_10 = np.asarray(settings["energy_spectrum"]["weights_10"], dtype=np.float32)
        weights_40 = np.asarray(settings["energy_spectrum"]["weights_40"], dtype=np.float32)
        self.weights = np.zeros_like(energies, dtype=np.float32)
        for i in range(len(energies)):
            self.weights[i] = np.interp(field_size, [3, 10, 40], [weights_3[i], weights_10[i], weights_40[i]])
        mu_w = mu_water(energies).astype(np.float32)

        spectrum_depth_res_cm = np.float32(0.2)
        max_spectrum_depth_cm = np.float32(30.0)
        spectrum_depths = np.arange(
            0.0,
            max_spectrum_depth_cm + spectrum_depth_res_cm * 0.5,
            spectrum_depth_res_cm,
            dtype=np.float32,
        )

        # Store phi (polar) angles and allocate depth-dependent kernel table
        self.phis = kernels[0].angles_centres
        theta_count = 16.0
        self.thetas = np.linspace(0, 360 - (360 / theta_count), int(theta_count), dtype=np.float32)
        self.omegas = kernels[0].omegas.astype(np.float32) / theta_count

        values_depth = np.zeros(
            (len(spectrum_depths), kernels[0].kernel.shape[0], kernels[0].kernel.shape[1]),
            dtype=np.float32,
        )

        # Compute spectrum-hardened kernels at each depth
        for depth_idx, depth_cm in enumerate(spectrum_depths):
            attenuated_weights = self.weights * np.exp(-mu_w * depth_cm)
            attenuated_weights /= attenuated_weights.sum()
            # Combine differential kernels spectrally
            combined_diff = np.zeros_like(kernels[0].kernel, dtype=np.float32)
            for i in range(len(energies)):
                combined_diff += kernels[i].kernel * attenuated_weights[i] * energies[i]
            combined_diff /= combined_diff.sum()
            values_depth[depth_idx] = combined_diff #/ theta_count

        # Compute cumulative kernels from differential
        values_depth_cum = np.cumsum(values_depth, axis=2)  # Cumsum along depth axis (axis 2)
        
        # Expose both differential and cumulative kernel tables
        self.values_depth_diff = values_depth.astype(np.float32)  # Differential kernels
        self.values_depth = values_depth_cum.astype(np.float32)   # Cumulative kernels
        self.values = self.values_depth[0]  # Legacy surface slice (cumulative at spectrum depth 0)

        # Store binning info
        self.n_depth_bins = np.int32(1192)
        self.kernel_depth_res_cm = np.float32(0.05)
        self.max_kernel_depth_cm = np.float32(59.6)
        self.ds_cm = np.float32(0.05)
        self.spectrum_depth_res_cm = spectrum_depth_res_cm
        self.max_spectrum_depth_cm = spectrum_depths[-1]
        self.n_spectrum_depth_bins = np.int32(len(spectrum_depths))
