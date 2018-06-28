import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from conehead.conehead import Source, beam_to_global, global_to_beam

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

# Create slab phantom
phantom = np.mgrid[-100:100:41j, -400:400:41j, -100:100:41j]


def line_plane_collision(plane_normal, plane_point, ray_direction, ray_point,
                         epsilon=1e-6):

    ndotu = plane_normal.dot(ray_direction)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = ray_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    psi = w + si * ray_direction + plane_point
    return psi


# Define plane
plane_normal = np.array([0, 1, 0])
plane_point = np.array([0, 0, 0])  # Any point on the plane

# Define ray
ray_point = source.position  # Any point along the ray

_, xlen, ylen, zlen = phantom.shape
phantom_blocked = np.zeros((xlen, ylen, zlen))

for i in range(xlen):
    for j in range(ylen):
        for k in range(zlen):
            voxel = phantom[:, i, j, k]
            ray_direction = voxel - source.position
            psi = line_plane_collision(
                plane_normal, plane_point, ray_direction, ray_point
            )
            phantom_blocked[i, j, k] = source.block_plane_values_interp(
               [psi[0], psi[2]]
            )
# print("intersection at", psi)
# print("voxel position", voxel)

f = plt.figure()
ax = plt.gca()
ax.imshow(
    np.flipud(phantom_blocked[20, :, :]),
    extent=[-102.5, 102.5, -410, 410],
    aspect='auto'
)
# Minor ticks
# ax.set_xticks(np.arange(-97.5, 100, 5), minor=True)
# ax.set_yticks(np.arange(-390, 400, 20), minor=True)
# ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
plt.show()
