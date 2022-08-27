import numpy as np


def dda_3d(direction, grid, current_voxel, voxel_size):

    step = np.zeros(3)
    step[0] = -1 if direction[0] < 0 else 1
    step[1] = -1 if direction[1] < 0 else 1
    step[2] = -1 if direction[2] < 0 else 1

    t = np.zeros(3)
    delta_t = np.zeros(3)
    big_number = 1000000000

    if direction[0] == 0.0:
        t[0] = big_number
        delta_t[0] = big_number
    else:
        t[0] = (voxel_size[0] / 2) / direction[0]
        delta_t[0] = voxel_size[0] / direction[0]
    if direction[1] == 0.0:
        t[1] = big_number
        delta_t[1] = big_number
    else:
        t[1] = (voxel_size[1] / 2) / direction[1]
        delta_t[1] = voxel_size[1] / direction[1]
    if direction[2] == 0.0:
        t[2] = big_number
        delta_t[2] = big_number
    else:
        t[2] = (voxel_size[2] / 2) / direction[2]
        delta_t[2] = voxel_size[2] / direction[2]

    xmax, ymax, zmax = grid.shape

    voxels_traversed = []
    intersection_t_values = []

    while (current_voxel[0] >= 0 and current_voxel[0] < xmax and
           current_voxel[1] >= 0 and current_voxel[1] < ymax and
           current_voxel[2] >= 0 and current_voxel[2] < zmax):

        voxels_traversed.append(np.copy(current_voxel))
        if t[0] < t[1]:
            if t[0] < t[2]:
                intersection_t_values.append(t[0])
                t[0] += delta_t[0]
                current_voxel[0] += step[0]
            else:
                intersection_t_values.append(t[2])
                t[2] += delta_t[2]
                current_voxel[2] += step[2]
        else:
            if t[1] < t[2]:
                intersection_t_values.append(t[1])
                t[1] += delta_t[1]
                current_voxel[1] += step[1]
            else:
                intersection_t_values.append(t[2])
                t[2] += delta_t[2]
                current_voxel[2] += step[2]

    return (voxels_traversed, intersection_t_values)


# source = np.array([0.0, 0.0, 0.0])
# direction = np.array([1/np.sqrt(14), 2/np.sqrt(14), 3/np.sqrt(14)])
# grid = np.ones((11, 11, 11))
# current_voxel = np.array([5, 5, 5])
# voxel_size = np.array([1.0, 1.0, 1.0])

# voxels, hits = dda_3d(source, direction, grid, current_voxel, voxel_size)
