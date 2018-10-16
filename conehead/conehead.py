import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from .geometry import (
    Transform, line_block_plane_collision,
    line_calc_limit_plane_collision, isocentre_plane_position
)
from .kernel import PolyenergeticKernel
from .convolve_c import convolve_c # pylint: disable=E0611
from .nist import mu_water

# Ignore annoying numpy FutureWarning, should be fixed by numpy 1.20
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Conehead:

    def calculate(self, source, block, phantom, settings):

        # Transform phantom to beam coords
        print("Transforming phantom to beam coords...")
        transform = Transform(source.position, source.rotation)
        phantom_beam = np.zeros_like(phantom.positions)
        _, xlen, ylen, zlen = phantom_beam.shape
        for x in tqdm(range(xlen)):
            for y in range(ylen):
                for z in range(zlen):
                    phantom_beam[:, x, y, z] = transform.global_to_beam(
                        phantom.positions[:, x, y, z],
                    )

        print("Interpolating phantom densities...")
        phantom_densities_interp = RegularGridInterpolator(
            (phantom_beam[0, :, 0, 0],
             phantom_beam[1, 0, :, 0],
             phantom_beam[2, 0, 0, :]),
            phantom.densities,
            method='nearest',
            bounds_error=False,
            fill_value=0
        )

        # Create dose grid (just the same size as the phantom for now)
        self.dose_grid_positions = np.copy(phantom_beam)
        self.dose_grid_dim = np.array([1, 1, 1], dtype=np.float64)  # cm

        # Perform hit testing to find which dose grid voxels are in the beam
        print("Performing hit-testing of dose grid voxels...")
        _, xlen, ylen, zlen = self.dose_grid_positions.shape
        dose_grid_blocked = np.zeros((xlen, ylen, zlen))
        dose_grid_OAD = np.zeros((xlen, ylen, zlen))
        for x in tqdm(range(xlen)):
            for y in range(ylen):
                for z in range(zlen):
                    position = self.dose_grid_positions[:, x, y, z]
                    position_iso = isocentre_plane_position(
                        position, source.SAD
                    )
                    dose_grid_blocked[x, y, z] = (
                        block.transmission(position_iso)
                    )
                    # Save off-axis distance (at iso plane) for later
                    dose_grid_OAD[x, y, z] = (
                        np.sqrt(np.sum(np.power(position_iso, 2)))
                    )

        # Calculate effective depths of dose grid voxels
        print("Calculating effective depths of dose grid voxels...")
        dose_grid_d_eff = np.zeros_like(dose_grid_blocked)
        xlen, ylen, zlen = dose_grid_d_eff.shape
        max_z = np.max(self.dose_grid_positions[2, :, :, :])
        print(max_z)
        for x in tqdm(range(xlen)):
            for y in range(ylen):
                for z in range(zlen):
                    if not dose_grid_blocked[x, y, z]:
                        continue
                    voxel = self.dose_grid_positions[:, x, y, z]
                    psi = line_calc_limit_plane_collision(
                        voxel, np.array([0, 0, max_z + 5.0])
                    )
                    dist = np.sqrt(np.sum(np.power(voxel - psi, 2)))
                    num_steps = np.floor(dist / settings['stepSize'])
                    xcoords = np.linspace(voxel[0], psi[0], num_steps)
                    ycoords = np.linspace(voxel[1], psi[1], num_steps)
                    zcoords = np.linspace(voxel[2], psi[2], num_steps)
                    dose_grid_d_eff[x, y, z] = np.sum(
                        phantom_densities_interp(
                            np.dstack((xcoords, ycoords, zcoords))
                        ) * settings['stepSize']
                    )

        # Calculate photon fluence at dose grid voxels
        print("Calculating fluence...")
        self.dose_grid_fluence = np.zeros_like(dose_grid_blocked)
        xlen, ylen, zlen = self.dose_grid_fluence.shape
        self.dose_grid_fluence = (
            settings['sPri'] * -source.SAD /
            self.dose_grid_positions[2, :, :, :] *
            dose_grid_blocked
        )

        # Calculate beam softening factor for dose grid voxels
        print("Calculating beam softening factor...")
        f_soften = np.ones_like(dose_grid_OAD)
        f_soften[dose_grid_OAD < settings['softLimit']] = 1 / (
            1 - settings['softRatio'] *
            dose_grid_OAD[dose_grid_OAD < settings['softLimit']]
        )

        # Calculate TERMA of dose grid voxels
        print("Calculating TERMA...")
        E = np.linspace(settings['eLow'], settings['eHigh'], settings['eNum'])
        spectrum_weights = source.weights(E)
        mu_w = mu_water(E)
        self.dose_grid_terma = np.zeros_like(dose_grid_blocked)
        xlen, ylen, zlen = self.dose_grid_terma.shape
        for x in tqdm(range(xlen)):
            for y in range(ylen):
                for z in range(zlen):
                    if not dose_grid_blocked[x, y, z]:
                        continue
                    self.dose_grid_terma[x, y, z] = (
                        np.sum(
                            spectrum_weights *
                            self.dose_grid_fluence[x, y, z] *
                            np.exp(
                                -mu_w * f_soften[x, y, z] *
                                dose_grid_d_eff[x, y, z]
                            ) * E * mu_w
                        )
                    )

        # Calculate dose of dose grid voxels
        print("Convolving kernel...")
        kernel = PolyenergeticKernel()
        dose_grid_dose = np.zeros_like(self.dose_grid_terma, dtype=np.float64)
        phis = np.array(
            sorted([p for p in kernel.cumulative.keys() if p != "radii"]),
            dtype=np.float64
        )
        thetas = np.linspace(0, 360, 12, endpoint=False, dtype=np.float64)
        convolve_c(
            self.dose_grid_terma,
            dose_grid_dose,
            self.dose_grid_dim,
            thetas,
            phis,
            kernel
        )
        self.dose_grid_dose = dose_grid_dose


    def plot(self):
        # Plotting for debug purposes
        print("Calculation complete. Now plotting...")

        # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        f1 = plt.figure()  # pylint: disable=W0612
        ax1 = plt.gca()
        im = ax1.imshow(  # pylint: disable=W0612
            np.rot90(self.dose_grid_fluence[:, 20, :]),
            extent=[-20.5, 20.5, -40.5, .5],
            aspect='equal',
            cmap='viridis'
        )
        # Minor ticks
        ax1.set_xticks(np.arange(-19.5, 20.0, 1.0), minor=True)
        ax1.set_yticks(np.arange(-39.5, 0, 1.0), minor=True)
        # ax1.grid(which="minor", color="#666666", linestyle='-', linewidth=1)
        ax1.set_title('Fluence')
        # plt.colorbar(im)

        f2 = plt.figure()  # pylint: disable=W0612
        ax2 = plt.gca()
        im = ax2.imshow(
            np.rot90(self.dose_grid_terma[:, 20, :]),
            extent=[-20.5, 20.5, -40.5, .5],
            aspect='equal',
            cmap='viridis'
        )
        # Minor ticks
        ax2.set_xticks(np.arange(-19.5, 20.0, 1.0), minor=True)
        ax2.set_yticks(np.arange(-39.5, 0, 1.0), minor=True)
        # ax2.grid(which="minor", color="#666666", linestyle='-', linewidth=1)
        ax2.set_title('TERMA')
        # plt.colorbar(im)

        f3 = plt.figure()  # pylint: disable=W0612
        ax3 = plt.gca()
        im = ax3.imshow(
            np.rot90(self.dose_grid_dose[:, 20, :]),
            extent=[-20.5, 20.5, -40.5, .5],
            aspect='equal',
            cmap='viridis'
        )
        # Minor ticks
        ax3.set_xticks(np.arange(-19.5, 20.0, 1.0), minor=True)
        ax3.set_yticks(np.arange(-39.5, 0, 1.0), minor=True)
        # ax3.grid(which="minor", color="#666666", linestyle='-', linewidth=1)
        ax3.set_title('Dose')
        # plt.colorbar(im)


        f6 = plt.figure()  # pylint: disable=W0612
        ax6 = plt.gca()
        im = ax6.imshow(
            self.dose_grid_dose[:, :, 10],
            extent=[-20.5, 20.5, -20.5, 20.5],
            aspect='equal',
            cmap='viridis'
        )
        # Minor ticks
        ax6.set_xticks(np.arange(-19.5, 20.0, 1.0), minor=True)
        ax6.set_yticks(np.arange(-19.5, 20.0, 1.0), minor=True)
        # ax6.grid(which="minor", color="#666666", linestyle='-', linewidth=1)
        ax6.set_title('Dose')
        # plt.colorbar(im)


        f4 = plt.figure()  # pylint: disable=W0612
        ax4 = plt.gca()
        ax4.plot(
            self.dose_grid_positions[0, :, 20, 38],
            (self.dose_grid_dose[:, 20, 38] /
             np.max(self.dose_grid_dose[:, 20, 38]) * 100),
            label='2 cm'
        )
        ax4.plot(
            self.dose_grid_positions[0, :, 20, 30],
            (self.dose_grid_dose[:, 20, 30] /
             np.max(self.dose_grid_dose[:, 20, 38]) * 100),
            label='10 cm'
        )
        ax4.plot(
            self.dose_grid_positions[0, :, 20, 10],
            (self.dose_grid_dose[:, 20, 10] /
             np.max(self.dose_grid_dose[:, 20, 38]) * 100),
            label='30 cm'
        )
        # ax4.plot(
        #     -self.dose_grid_positions[2, 20, 20, :] - 100,
        #     (self.dose_grid_terma[20, 20, :] /
        #      np.max(self.dose_grid_terma[20, 20, :]) * 100),
        #     label='TERMA'
        # )
        # ax4.plot(
        #     -self.dose_grid_positions[2, 20, 20, :] - 100,
        #     (self.dose_grid_dose[20, 20, :] /
        #      np.max(self.dose_grid_dose[20, 20, :]) * 100),
        #     label='Dose'
        # )
        ax4.set_xlim([-15, 15])
        ax4.set_ylim([0, 110])
        ax4.set_title("Profiles")
        ax4.set_xlabel("Position [cm]")
        ax4.set_ylabel("Relative Value [%]")
        x0, x1 = ax4.get_xlim()
        y0, y1 = ax4.get_ylim()
        ax4.set_aspect(abs(x1-x0)/abs(y1-y0))
        ax4.legend()

        f5 = plt.figure()  # pylint: disable=W0612
        ax5 = plt.gca()
        # ax5.plot(
        #     -self.dose_grid_positions[2, 20, 20, :] - 100,
        #     (self.dose_grid_fluence[20, 20, :] /
        #      np.max(self.dose_grid_fluence[20, 20, :]) * 100),
        #     label='Fluence'
        # )
        # ax5.plot(
        #     -self.dose_grid_positions[2, 20, 20, :] - 100,
        #     (self.dose_grid_terma[20, 20, :] /
        #      np.max(self.dose_grid_terma[20, 20, :]) * 100),
        #     label='TERMA'
        # )
        ax5.plot(
            -self.dose_grid_positions[2, 20, 20, :] - 100,
            (self.dose_grid_dose[20, 20, :] /
             np.max(self.dose_grid_dose[20, 20, :]) * 100)
            # label='Dose'
        )
        ax5.set_xlim([0, 40.0])
        ax5.set_ylim([0, 100])
        ax5.set_title("PDD")
        ax5.set_xlabel("Depth [cm]")
        ax5.set_ylabel("Relative Value [%]")
        x0, x1 = ax5.get_xlim()
        y0, y1 = ax5.get_ylim()
        ax5.set_aspect(abs(x1-x0)/abs(y1-y0))
        # ax5.legend()

        plt.show()
