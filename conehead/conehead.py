import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from conehead.source import Source
from conehead.geometry import (
    global_to_beam, line_block_plane_collision, line_calc_limit_plane_collision
)
import conehead.nist
import conehead.clinac_6MV_spectrum


class Conehead:

    def calculate(self, settings):

        # Create source
        source = Source()
        source.gantry(settings['gantryAngle'])
        source.collimator(settings['collimatorAngle'])

        # Set 100 mm x 100 mm collimator opening
        block_plane_locations = np.mgrid[-20:20:40j, -20:20:40j]
        block_plane_values = np.zeros((400, 400))
        block_plane_values[150:250, 150:250] = 1
        source.block_plane_values = block_plane_values
        source.block_plane_values_interp = RegularGridInterpolator(
            (np.linspace(-20, 20, 400), np.linspace(-20, 20, 400)),
            block_plane_values,
            method='nearest',
            bounds_error=False,
            fill_value=0
        )

        # Create slab phantom in global coords
        print("Creating phantom...")
        phantom_positions = np.mgrid[-20:20:41j, -20:20:41j, -40:0:41j]
        _, xlen, ylen, zlen = phantom_positions.shape
        phantom_densities = np.ones((xlen, ylen, zlen))  # Water
        # phantom_densities[15:26, 15:26, 15:26] = 4  # Higher density feature

        # Transform phantom to beam coords
        print("Transforming phantom to beam coords...")
        phantom_beam = np.zeros_like(phantom_positions)
        for x in tqdm(range(xlen)):
            for y in range(ylen):
                for z in range(zlen):
                    phantom_beam[:, x, y, z] = global_to_beam(
                        phantom_positions[:, x, y, z],
                        source.position,
                        source.rotation
                    )

        print("Interpolating phantom densities...")
        phantom_densities_interp = RegularGridInterpolator(
            (phantom_beam[0, :, 0, 0],
             phantom_beam[1, 0, :, 0],
             phantom_beam[2, 0, 0, :]),
            phantom_densities,
            method='nearest',
            bounds_error=False,
            fill_value=0
        )

        # Create dose grid (just the same size as the phantom for now)
        self.dose_grid_positions = np.copy(phantom_beam)

        # Perform hit testing to find which dose grid voxels are in the beam
        print("Performing hit-testing of dose grid voxels...")
        _, xlen, ylen, zlen = self.dose_grid_positions.shape
        dose_grid_blocked = np.zeros((xlen, ylen, zlen))
        dose_grid_OAD = np.zeros((xlen, ylen, zlen))
        for x in tqdm(range(xlen)):
            for y in range(ylen):
                for z in range(zlen):
                    voxel = self.dose_grid_positions[:, x, y, z]
                    psi = line_block_plane_collision(voxel)
                    dose_grid_blocked[x, y, z] = (
                        source.block_plane_values_interp([psi[0], psi[1]])
                    )
                    # Save off-axis distance (at iso plane) for later
                    dose_grid_OAD[x, y, z] = (
                        euclidean(np.array([0, 0, -settings['SAD']]), psi)
                    )

        # Calculate effective depths of dose grid voxels
        print("Calculating effective depths of dose grid voxels...")
        dose_grid_d_eff = np.zeros_like(dose_grid_blocked)
        xlen, ylen, zlen = dose_grid_d_eff.shape
        for x in tqdm(range(xlen)):
            for y in range(ylen):
                for z in range(zlen):
                    voxel = self.dose_grid_positions[:, x, y, z]
                    psi = line_calc_limit_plane_collision(voxel)
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
            settings['sPri'] * -settings['SAD'] /
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
        spectrum_weights = conehead.clinac_6MV_spectrum.weights(E)
        mu_water = conehead.nist.mu_water(E)
        self.dose_grid_terma = np.zeros_like(dose_grid_blocked)
        xlen, ylen, zlen = self.dose_grid_terma.shape
        for x in tqdm(range(xlen)):
            for y in range(ylen):
                for z in range(zlen):
                    self.dose_grid_terma[x, y, z] = (
                        np.sum(
                            spectrum_weights *
                            self.dose_grid_fluence[x, y, z] *
                            np.exp(
                                -mu_water * f_soften[x, y, z] *
                                dose_grid_d_eff[x, y, z]
                            ) * E * mu_water
                        )
                    )

    def plot(self):
        # Plotting for debug purposes
        print("Calculation complete. Now plotting...")
        f1 = plt.figure()
        ax = plt.gca()
        im = ax.imshow(
            np.rot90(self.dose_grid_terma[:, 20, :]),
            extent=[-20.5, 20.5, -40.5, .5],
            aspect='equal'
        )
        # Minor ticks
        ax.set_xticks(np.arange(-19.5, 20.0, 1.0), minor=True)
        ax.set_yticks(np.arange(-39.5, 0, 1.0), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        plt.colorbar(im)

        f2 = plt.figure()
        ax2 = plt.gca()
        ax2.plot(
            -self.dose_grid_positions[2, 20, 20, :] - 100,
            (self.dose_grid_fluence[20, 20, :] /
             np.max(self.dose_grid_fluence[20, 20, :]) * 100),
            label='Fluence'
        )
        ax2.plot(
            -self.dose_grid_positions[2, 20, 20, :] - 100,
            (self.dose_grid_terma[20, 20, :] /
             np.max(self.dose_grid_terma[20, 20, :]) * 100),
            label='TERMA'
        )
        ax2.set_xlim([0, 40.0])
        ax2.set_ylim([0, 100])
        ax2.set_title("Central Axis Quantites")
        ax2.set_xlabel("Depth [cm]")
        ax2.set_ylabel("Relative Value [%]")
        ax2.legend()
        plt.show()
