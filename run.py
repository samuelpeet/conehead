import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from conehead.source import Source
from conehead.geometry import (
    beam_to_global, global_to_beam, line_plane_collision
)


# Create source
source = Source()
source.gantry(270)
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
phantom = np.mgrid[-200:200:41j, -200:200:41j, -200:200:41j]

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

# Perform hit testing to find which voxels are in the beam
_, xlen, ylen, zlen = phantom_beam.shape
phantom_blocked = np.zeros((xlen, ylen, zlen))
for x in range(xlen):
    for y in range(ylen):
        for z in range(zlen):
            voxel = phantom_beam[:, x, y, z]
            psi = line_plane_collision(voxel)
            phantom_blocked[x, y, z] = source.block_plane_values_interp(
               [psi[0], psi[1]]
            ) + np.random.random_sample()*0.2

# Plotting for debug purposes
f = plt.figure()
ax = plt.gca()
ax.imshow(
    np.rot90(phantom_blocked[:, 20, :]),
    extent=[-205, 205, -205, 205],
    aspect='equal'
)
# # Minor ticks
# ax.set_xticks(np.arange(-97.5, 100, 5), minor=True)
# ax.set_yticks(np.arange(-390, 400, 20), minor=True)
# ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
plt.show()
