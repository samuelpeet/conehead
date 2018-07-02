import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from conehead.source import Source
from conehead.geometry import (
    global_to_beam, line_plane_collision
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
phantom = np.mgrid[-200:200:41j, -200:200:41j, -400:0:41j]

# Transform phantom to beam coords
phantom_beam = np.zeros_like(phantom)
for x in range(41):
    for y in range(41):
        for z in range(41):
            phantom_beam[:, x, y, z] = global_to_beam(
                phantom[:, x, y, z],
                source.position,
                source.rotation
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
            psi = line_plane_collision(voxel)
            dose_grid_blocked[x, y, z] = source.block_plane_values_interp(
               [psi[0], psi[1]]
            )

# Calculate photon fluence
S_pri = 1.0  # Primary source strength (photons/mm^2)
dose_grid_fluence = np.zeros_like(dose_grid_blocked)
for x in range(xlen):
    for y in range(ylen):
        for z in range(zlen):
            dose_grid_fluence[x, y, z] = S_pri
            dose_grid_fluence[x, y, z] *= (-1000 / dose_grid_positions[2, x, y, z])
            dose_grid_fluence[x, y, z] *= dose_grid_blocked[x, y, z]
            # dose_grid_fluence[x, y, z] += np.random.random_sample()*0.1

# Plotting for debug purposes
f = plt.figure()
ax = plt.gca()
ax.imshow(
    np.rot90(dose_grid_fluence[:, 20, :]),
    extent=[-205, 205, -405, 5],
    aspect='equal'
)
# # Minor ticks
# ax.set_xticks(np.arange(-97.5, 100, 5), minor=True)
# ax.set_yticks(np.arange(-390, 400, 20), minor=True)
# ax.grid(which="minor", color="w", linestyle='-', linewidth=1)

f = plt.figure()
ax2 = plt.gca()
ax2.plot((dose_grid_positions[2, 20, 20, :] + 1000) * -1, dose_grid_fluence[20, 20, :] / np.max(dose_grid_fluence[20, 20, :]) * 100, label='Fluence')
ax2.set_xlim([0, 400])
ax2.set_ylim([0, 100])
ax2.set_title("Central Axis Quantites")
ax2.set_xlabel("Depth [mm]")
ax2.set_ylabel("Relative Value [%]")
ax2.legend()
plt.show()
