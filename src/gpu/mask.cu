#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

/**
 * @brief CUDA kernel that builds a binary mask indicating voxels with nearby TERMA.
 *
 * This __global__ kernel examines a cubic neighborhood along the eight diagonal
 * directions around each voxel and marks the central voxel active (1.0f) if
 * any neighbor within the specified physical radius has a TERMA value >= the
 * provided threshold. The mask is intended to accelerate downstream dose
 * convolution by skipping voxels with negligible nearby TERMA.
 *
 * Implementation notes:
 *  - The physical search radius `max_distance_cm` is converted to a maximum
 *    integer voxel radius using the smallest voxel dimension (min(dx,dy,dz)).
 *  - The search probes only the eight diagonal rays (combinations of ±x, ±y, ±z)
 *    to keep runtime low while conservatively detecting nearby high-TERMA voxels.
 *
 * @param[out] mask_grid         Device pointer to flattened float array (nx*ny*nz). Values will be 1.0f (active) or 0.0f (inactive).
 * @param[in]  terma_grid        Device pointer to flattened TERMA float array (nx*ny*nz).
 * @param[in]  num_voxels        Device pointer to int[3] containing {nx,ny,nz}.
 * @param[in]  resolution        Device pointer to float[3] voxel sizes (dx,dy,dz) in same units as max_distance_cm.
 * @param[in]  max_distance_cm   Physical search radius (cm) used to determine neighborhood.
 * @param[in]  terma_threshold   TERMA threshold; any neighbor with terma >= this marks the voxel as active.
 *
 * @note The kernel uses row-major flattening (idx = x + y*nx + z*nx*ny) and
 *       expects the host to launch a grid/block configuration that covers
 *       all voxels. The search is conservative but inexpensive.
 */
__global__ void mask(float* mask_grid,
    float* terma_grid,
    int* num_voxels,
    float* resolution,
    float max_distance_cm,
    float terma_threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = num_voxels[0];
    int ny = num_voxels[1];
    int nz = num_voxels[2];

    int max_voxels_distance = (int)(max_distance_cm / fmin(resolution[0], fmin(resolution[1], resolution[2])));

    if (x < nx && y < ny && z < nz) {
        int idx = x + y * nx + z * nx * ny;

        // We check 8 directions (±x, ±y, ±z)
        for (int ix = -1; ix <= 1; ix = ix + 2) // Just alternate from negative to positive 1
        {
            for (int iy = -1; iy <= 1; iy = iy + 2) {
                for (int iz = -1; iz <= 1; iz = iz + 2) {
                    for (int v = 0; v <= max_voxels_distance; v++) {
                        int cx = x + ix * v;
                        int cy = y + iy * v;
                        int cz = z + iz * v;

                        // Ensure neighbor indices are within bounds
                        if (cx >= 0 && cx < nx && cy >= 0 && cy < ny && cz >= 0 && cz < nz) {
                            int n_idx = cx + cy * nx + cz * nx * ny;
                            if (terma_grid[n_idx] >= terma_threshold) {
                                mask_grid[idx] = 1.0f;
                                return;
                            }
                        }
                    }
                }
            }
        }
        mask_grid[idx] = 0.0f; // No neighbors within threshold
    }
}
/**
 * @brief Python-visible host wrapper that builds the TERMA-based mask.
 *
 * This function is intended to be exposed via pybind11. It accepts NumPy
 * buffers for the output mask and input TERMA/geometry arrays, transfers
 * data to the device, launches the `mask` kernel, and copies the resulting
 * mask back into the provided output array.
 *
 * All NumPy arrays must be C-contiguous and use dtype float32 (except
 * `num_voxels` which should be int32). The output `mask_grid` is written in
 * place and will contain 0.0f/1.0f values after return.
 *
 * @param[in,out] mask_grid      NumPy array (float32, flattened) preallocated to hold nx*ny*nz elements.
 * @param[in]     terma_grid     NumPy array (float32, flattened) containing TERMA values.
 * @param[in]     num_voxels     NumPy array (int32, shape=(3,)) holding [nx,ny,nz].
 * @param[in]     resolution     NumPy array (float32, shape=(3,)) voxel sizes (dx,dy,dz).
 * @param[in]     max_distance_cm Float: physical search radius in same units as resolution.
 * @param[in]     terma_threshold Float: TERMA threshold used to mark nearby voxels active.
 *
 * @note This wrapper performs synchronous host-to-device and device-to-host
 *       copies on the default stream. For repeated calls consider creating
 *       persistent device buffers to avoid allocation overhead.
 */
void map_mask(pybind11::array_t<float> mask_grid,
    pybind11::array_t<float> terma_grid,
    pybind11::array_t<int> num_voxels,
    pybind11::array_t<float> resolution,
    float max_distance_cm,
    float terma_threshold)
{
    pybind11::buffer_info mask_grid_info = mask_grid.request();
    pybind11::buffer_info terma_grid_info = terma_grid.request();
    pybind11::buffer_info num_voxels_info = num_voxels.request();
    pybind11::buffer_info resolution_info = resolution.request();

    float* mask_grid_ptr = reinterpret_cast<float*>(mask_grid_info.ptr);
    float* terma_grid_ptr = reinterpret_cast<float*>(terma_grid_info.ptr);
    int* num_voxels_ptr = reinterpret_cast<int*>(num_voxels_info.ptr);
    float* resolution_ptr = reinterpret_cast<float*>(resolution_info.ptr);

    // Allocate device memory and copy inputs
    float *d_mask_grid, *d_terma_grid, *d_resolution;
    int* d_num_voxels;
    cudaMalloc(&d_mask_grid, mask_grid_info.size * sizeof(float));
    cudaMalloc(&d_terma_grid, terma_grid_info.size * sizeof(float));
    cudaMalloc(&d_num_voxels, num_voxels_info.size * sizeof(int));
    cudaMalloc(&d_resolution, resolution_info.size * sizeof(float));
    cudaMemcpy(d_mask_grid, mask_grid_ptr, mask_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_terma_grid, terma_grid_ptr, terma_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_voxels, num_voxels_ptr, num_voxels_info.size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_resolution, resolution_ptr, resolution_info.size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(16, 4, 4);
    dim3 dimGrid((num_voxels_ptr[0] + dimBlock.x - 1) / dimBlock.x,
        (num_voxels_ptr[1] + dimBlock.y - 1) / dimBlock.y,
        (num_voxels_ptr[2] + dimBlock.z - 1) / dimBlock.z);

    mask<<<dimGrid, dimBlock>>>(d_mask_grid, d_terma_grid, d_num_voxels,
        d_resolution, max_distance_cm, terma_threshold);

    // Copy result back to host
    cudaMemcpy(mask_grid_ptr, d_mask_grid,
        mask_grid_info.size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_mask_grid);
    cudaFree(d_terma_grid);
    cudaFree(d_num_voxels);
    cudaFree(d_resolution);
}
