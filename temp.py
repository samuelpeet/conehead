#%%
import numpy as np
import os; os.environ["NUMBA_ENABLE_CUDASIM"] = "0"; os.environ["NUMBA_CUDA_DEBUGINFO"] = "0";
import numba
from numba import cuda
import conehead.dda_3d
import time
import math



# grid = np.zeros((64, 64, 64))
# d_eff_cpu = np.zeros_like(grid)
# direction = [1, 0, 0]
# voxel_size = [1, 1, 1]

# start = time.time()
# for i in range(grid.shape[0]):
#     for j in range(grid.shape[1]):
#         for k in range(grid.shape[2]):
#             current_voxel = [i, j, k]
#             voxels_traversed, intersection_t_values = conehead.dda_3d.dda_3d(direction, grid, current_voxel, voxel_size)  # type: ignore
#             d_eff_cpu[i, j, k] = np.sum(intersection_t_values)
# end = time.time()
# print(f"{end - start:.2f} s")
# print(d_eff)


# %%
@cuda.jit
def dda_3d_cuda(direction, grid, voxel_size, d_eff):

    x, y, z = cuda.grid(3)
    current_voxel = cuda.local.array(3, numba.int32)
    current_voxel[0] = x
    current_voxel[1] = y
    current_voxel[2] = z
    if x < grid.shape[0] and y < grid.shape[1] and z < grid.shape[2]:

        # print(x, y, z)

        step = cuda.local.array(3, numba.int32)
        step[0] = -1 if direction[0] < 0 else 1
        step[1] = -1 if direction[1] < 0 else 1
        step[2] = -1 if direction[2] < 0 else 1
        # print(step[0], step[1], step[2])

        t = cuda.local.array(3, numba.float32)
        t[0] = 0.0
        t[1] = 0.0
        t[2] = 0.0
        

        delta_t = cuda.local.array(3, numba.float32)
        delta_t[0] = 0.0
        delta_t[1] = 0.0
        delta_t[2] = 0.0
        big_number = 1000000000.0
        

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


        # print(x, y, z, t[0], t[1], t[2], delta_t[0], delta_t[1], delta_t[2])


        xmax = grid.shape[0]
        ymax = grid.shape[1]
        zmax = grid.shape[2]

        # voxels_traversed = cuda.local.array(100,)
        max_array_length = 444  # Longest diagonal of 256 x 256 x 256 grid 
        intersection_t_values = cuda.local.array(max_array_length, numba.float32)
        intersection_t_values_count = 0

        while (current_voxel[0] >= 0 and current_voxel[0] < xmax and
            current_voxel[1] >= 0 and current_voxel[1] < ymax and
            current_voxel[2] >= 0 and current_voxel[2] < zmax):

            # voxels_traversed.append(np.copy(current_voxel))
            if t[0] < t[1]:
                if t[0] < t[2]:
                    intersection_t_values[intersection_t_values_count] = t[0]
                    intersection_t_values_count += 1
                    t[0] += delta_t[0]
                    current_voxel[0] += step[0]
                else:
                    intersection_t_values[intersection_t_values_count] = t[2]
                    intersection_t_values_count += 1
                    t[2] += delta_t[2]
                    current_voxel[2] += step[2]
            else:
                if t[1] < t[2]:
                    intersection_t_values[intersection_t_values_count] = t[1]
                    intersection_t_values_count += 1
                    t[1] += delta_t[1]
                    current_voxel[1] += step[1]
                else:
                    intersection_t_values[intersection_t_values_count] = t[2]
                    intersection_t_values_count += 1
                    t[2] += delta_t[2]
                    current_voxel[2] += step[2]

        # print(x, y, z, t[0], t[1], t[2], delta_t[0], delta_t[1], delta_t[2], intersection_t_values.size)

        for i in range(len(intersection_t_values)):
            d_eff[x, y, z] += intersection_t_values[i]


grid = np.zeros((256, 256, 256), dtype=np.float32)
d_eff = np.zeros_like(grid, dtype=np.float32)
direction = [np.float32(1), np.float32(0), np.float32(0)]
voxel_size = [np.float32(1), np.float32(1), np.float32(1)]

grid_device = cuda.to_device(grid)
d_eff_device = cuda.to_device(d_eff)
direction_device = cuda.to_device(direction)
voxel_size_device = cuda.to_device(voxel_size)
 
threadsperblock = (8, 8, 8)
blockspergrid_x = math.ceil(grid.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(grid.shape[1] / threadsperblock[1])
blockspergrid_z = math.ceil(grid.shape[2] / threadsperblock[2])
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

print(threadsperblock)
print(blockspergrid)
print(grid.shape)

start = time.time()
dda_3d_cuda[blockspergrid, threadsperblock](direction_device, grid_device, voxel_size_device, d_eff_device)
d_eff_cuda = d_eff_device.copy_to_host()
end = time.time()
print(f"{end - start:.2f} s")


# print(d_eff_cpu)
# print(d_eff_cuda)

# assert (np.array_equal(d_eff_cpu, d_eff_cuda))
# print(d_eff)

# %%


