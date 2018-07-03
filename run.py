import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from conehead.source import Source
from conehead.geometry import (
    global_to_beam, line_block_plane_collision, line_calc_limit_plane_collision
)


# Create source
source = Source()
source.gantry(0)
source.collimator(0)

# Set 100 mm x 100 mm collimator opening
block_plane_locations = np.mgrid[-200:200:400j, -200:200:400j]
block_plane_values = np.zeros((400, 400))
block_plane_values[150:250, 150:250, ] = 1
source.block_plane_values = block_plane_values
source.block_plane_values_interp = RegularGridInterpolator(
    (np.linspace(-200, 200, 400), np.linspace(-200, 200, 400)),
    block_plane_values,
    method='nearest',
    bounds_error=False,
    fill_value=0
)

# Create slab phantom in global coords
phantom_positions = np.mgrid[-200:200:41j, -200:200:41j, -400:0:41j]
_, xlen, ylen, zlen = phantom_positions.shape
phantom_densities = np.ones((xlen, ylen, zlen))  # Water
phantom_densities[15:26, 15:26, 15:26] = 4  # Higher density feature

# Transform phantom to beam coords
phantom_beam = np.zeros_like(phantom_positions)
for x in range(41):
    for y in range(41):
        for z in range(41):
            phantom_beam[:, x, y, z] = global_to_beam(
                phantom_positions[:, x, y, z],
                source.position,
                source.rotation
            )
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
dose_grid_positions = np.copy(phantom_beam)

# Perform hit testing to find which dose grid voxels are in the beam
_, xlen, ylen, zlen = dose_grid_positions.shape
dose_grid_blocked = np.zeros((xlen, ylen, zlen))
for x in range(xlen):
    for y in range(ylen):
        for z in range(zlen):
            voxel = dose_grid_positions[:, x, y, z]
            psi = line_block_plane_collision(voxel)
            dose_grid_blocked[x, y, z] = source.block_plane_values_interp(
               [psi[0], psi[1]]
            )

# Calculate photon fluence
S_pri = 1.0  # Primary source strength (photons/mm^2)
dose_grid_fluence = np.zeros_like(dose_grid_blocked)
xlen, ylen, zlen = dose_grid_fluence.shape
for x in range(xlen):
    for y in range(ylen):
        for z in range(zlen):
            dose_grid_fluence[x, y, z] = S_pri
            dose_grid_fluence[x, y, z] *= (
                -1000 / dose_grid_positions[2, x, y, z]
            )
            dose_grid_fluence[x, y, z] *= dose_grid_blocked[x, y, z]
            # dose_grid_fluence[x, y, z] += np.random.random_sample()*0.1

# Calculate effective depths of dose grid voxels
step_size = 1  # mm
dose_grid_d_eff = np.zeros_like(dose_grid_blocked)
xlen, ylen, zlen = dose_grid_d_eff.shape
for x in range(xlen):
    for y in range(ylen):
        for z in range(zlen):
            voxel = dose_grid_positions[:, x, y, z]
            psi = line_calc_limit_plane_collision(voxel)
            dist = np.sqrt(np.sum(np.power(voxel - psi, 2)))
            num_steps = np.floor(dist / step_size)
            xcoords = np.linspace(voxel[0], psi[0], num_steps)
            ycoords = np.linspace(voxel[1], psi[1], num_steps)
            zcoords = np.linspace(voxel[2], psi[2], num_steps)
            dose_grid_d_eff[x, y, z] = np.sum(
                phantom_densities_interp(
                    np.dstack((xcoords, ycoords, zcoords))
                ) * step_size
            )

# Plotting for debug purposes
f1 = plt.figure()
ax = plt.gca()
im = ax.imshow(
    np.rot90(dose_grid_d_eff[:, 20, :]),
    extent=[-205, 205, -405, 5],
    aspect='equal'
)
# Minor ticks
ax.set_xticks(np.arange(-195, 200, 10), minor=True)
ax.set_yticks(np.arange(-395, 0, 10), minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
plt.colorbar(im)

f2 = plt.figure()
ax2 = plt.gca()
ax2.plot(
    (dose_grid_positions[2, 20, 20, :] + 1000) * -1, 
    dose_grid_fluence[20, 20, :] / np.max(dose_grid_fluence[20, 20, :]) * 100,
    label='Fluence'
)
ax2.set_xlim([0, 400])
ax2.set_ylim([0, 100])
ax2.set_title("Central Axis Quantites")
ax2.set_xlabel("Depth [mm]")
ax2.set_ylabel("Relative Value [%]")
ax2.legend()
plt.show()
