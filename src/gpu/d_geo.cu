#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

/**
 * @brief CUDA kernel that computes per-voxel geometric distance to a point source.
 *
 * This __global__ kernel assigns one CUDA thread per voxel. Each thread
 * computes the Euclidean distance between the provided ``source_position``
 * and the centre of its voxel and stores the result in the flattened
 * output array ``d_geo_grid`` at index ``idx = x + y*nx + z*nx*ny``.
 *
 * @param[out] d_geo_grid Device pointer to a flattened float array of length
 *                        nx*ny*nz where per-voxel distances are written.
 * @param[in]  num_voxels Device pointer to an int[3] containing {nx, ny, nz}.
 * @param[in]  corner     Device pointer to float[3] world-space corner
 *                        coordinates (minimum x,y,z) of the grid.
 * @param[in]  resolution Device pointer to float[3] voxel sizes (dx, dy, dz).
 * @param[in]  source_position Device pointer to float[3] describing the
 *                        world-space position of the point source.
 *
 * @note The kernel uses simple row-major flattening and expects the host to
 *       launch a grid/block configuration large enough to cover all voxels.
 */
__global__ void d_geo(float* d_geo_grid,
    int* num_voxels,
    float* corner,
    float* resolution,
    float* source_position)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = num_voxels[0];
    int ny = num_voxels[1];
    int nz = num_voxels[2];

    if (x < nx && y < ny && z < nz) {
        float3 corner_f3 = make_float3(corner[0], corner[1], corner[2]);
        float3 resolution_f3 = make_float3(resolution[0], resolution[1], resolution[2]);
        float3 source_position_f3 = make_float3(source_position[0], source_position[1], source_position[2]);
        float3 position_f3 = make_float3(0.0f, 0.0f, 0.0f);
        float3 ray_direction_f3 = make_float3(0.0f, 0.0f, 0.0f);

        // Get voxel position
        position_f3.x = corner_f3.x + resolution_f3.x * (x + 0.5);
        position_f3.y = corner_f3.y + resolution_f3.y * (y + 0.5);
        position_f3.z = corner_f3.z + resolution_f3.z * (z + 0.5);

        // Determine direction/distance to source
        ray_direction_f3.x = source_position_f3.x - position_f3.x;
        ray_direction_f3.y = source_position_f3.y - position_f3.y;
        ray_direction_f3.z = source_position_f3.z - position_f3.z;
        float mag = sqrt(ray_direction_f3.x * ray_direction_f3.x + ray_direction_f3.y * ray_direction_f3.y + ray_direction_f3.z * ray_direction_f3.z);
        int idx = x + y * nx + z * nx * ny;
        d_geo_grid[idx] = mag;
    }
}

/**
 * @brief Map NumPy buffers to device memory, launch the d_geo kernel, and
 *        write per-voxel Euclidean distances back into the provided array.
 *
 * This is the Python-facing host wrapper exposed via pybind11 as ``d_geo``.
 * It performs the following steps:
 *  - Validates and obtains raw pointers from the provided NumPy buffers.
 *  - Allocates temporary device memory and copies inputs to the GPU.
 *  - Launches the CUDA kernel ``d_geo`` to compute distances.
 *  - Copies the result back into ``d_geo_grid`` and frees device memory.
 *
 * @note All NumPy arrays passed to this function must be C-contiguous and use
 *       the exact dtypes specified below. Passing mismatched dtypes or
 *       non-contiguous arrays may lead to incorrect results or crashes.
 *
 * @param[in,out] d_geo_grid      numpy.ndarray (float32, shape=(nx,ny,nz))
 *      Preallocated output buffer. On return this array contains the Euclidean
 *      distance from ``source_position`` to the centre of each voxel.
 *      The wrapper treats the buffer as a flattened 1D array of length
 *      nx*ny*nz.
 * @param[in] num_voxels          numpy.ndarray (int32, shape=(3,))
 *      Integer array holding grid dimensions in the order [nx, ny, nz].
 * @param[in] corner              numpy.ndarray (float32, shape=(3,))
 *      World-space coordinates of the grid corner (minimum x,y,z).
 * @param[in] resolution          numpy.ndarray (float32, shape=(3,))
 *      Voxel sizes (dx, dy, dz) in the same units as ``corner``.
 * @param[in] source_position     numpy.ndarray (float32, shape=(3,))
 *      World-space coordinates of the point source.
 *
 * @par Threading / Streams
 * The wrapper performs synchronous cudaMemcpy calls and launches the kernel on
 * the default stream. For overlapping transfers and compute, consider
 * providing an asynchronous API that accepts a cudaStream_t and uses pinned
 * host memory.
 *
 * @par Example (Python)
 * @code{.py}
 * import numpy as np
 * import conehead_gpu as gpu
 * nx,ny,nz = 201,201,201
 * d_geo = np.zeros((nx,ny,nz), dtype=np.float32)
 * num_voxels = np.array([nx,ny,nz], dtype=np.int32)
 * corner = np.array([-20.1, 0.0, -20.1], dtype=np.float32)
 * resolution = np.array([0.2,0.2,0.2], dtype=np.float32)
 * source_position = np.array([0.0, -100.0, 0.0], dtype=np.float32)
 * gpu.d_geo(d_geo, num_voxels, corner, resolution, source_position)
 * @endcode
 */
void map_d_geo(pybind11::array_t<float> d_geo_grid, pybind11::array_t<int> num_voxels,
    pybind11::array_t<float> corner, pybind11::array_t<float> resolution,
    pybind11::array_t<float> source_position)
{
    pybind11::buffer_info d_geo_info = d_geo_grid.request();
    pybind11::buffer_info num_voxels_info = num_voxels.request();
    pybind11::buffer_info corner_info = corner.request();
    pybind11::buffer_info resolution_info = resolution.request();
    pybind11::buffer_info source_position_info = source_position.request();

    float* d_geo_ptr = reinterpret_cast<float*>(d_geo_info.ptr);
    int* num_voxels_ptr = reinterpret_cast<int*>(num_voxels_info.ptr);
    float* corner_ptr = reinterpret_cast<float*>(corner_info.ptr);
    float* resolution_ptr = reinterpret_cast<float*>(resolution_info.ptr);
    float* source_position_ptr = reinterpret_cast<float*>(source_position_info.ptr);

    // Allocate device memory and copy inputs
    float *d_d_geo_grid, *d_corner, *d_resolution, *d_source_position;
    int* d_num_voxels;
    cudaMalloc(&d_d_geo_grid, d_geo_info.size * sizeof(float));
    cudaMalloc(&d_num_voxels, num_voxels_info.size * sizeof(int));
    cudaMalloc(&d_corner, corner_info.size * sizeof(float));
    cudaMalloc(&d_resolution, resolution_info.size * sizeof(float));
    cudaMalloc(&d_source_position, source_position_info.size * sizeof(float));
    cudaMemcpy(d_d_geo_grid, d_geo_ptr, d_geo_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_voxels, num_voxels_ptr, num_voxels_info.size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_corner, corner_ptr, corner_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_resolution, resolution_ptr, resolution_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_position, source_position_ptr, source_position_info.size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(16, 4, 4);
    dim3 dimGrid((num_voxels_ptr[0] + dimBlock.x - 1) / dimBlock.x,
        (num_voxels_ptr[1] + dimBlock.y - 1) / dimBlock.y,
        (num_voxels_ptr[2] + dimBlock.z - 1) / dimBlock.z);

    d_geo<<<dimGrid, dimBlock>>>(d_d_geo_grid, d_num_voxels, d_corner,
        d_resolution, d_source_position);

    // Copy result back to host
    cudaMemcpy(d_geo_ptr, d_d_geo_grid,
        d_geo_info.size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_d_geo_grid);
    cudaFree(d_num_voxels);
    cudaFree(d_corner);
    cudaFree(d_resolution);
    cudaFree(d_source_position);
}
