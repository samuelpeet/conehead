#include "device_functions.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

/**
 * @brief CUDA kernel that computes the off-axis distance (OAD) per voxel.
 *
 * This __global__ kernel assigns one CUDA thread per voxel. For each voxel
 * the thread computes the voxel centre in world coordinates, projects the
 * point onto the source fluence plane using the provided source basis
 * vectors, converts the projected point into source-local coordinates and
 * stores the radial off-axis distance sqrt(x^2 + z^2) into the flattened
 * output array ``oad_grid`` at index ``idx = x + y*nx + z*nx*ny``.
 *
 * The projection uses a ray-plane intersection helper (``line_plane_collision``)
 * and the source-local coordinates are computed with the device dot helpers.
 *
 * @param[out] oad_grid         Device pointer to a flattened float array of
 *                             length nx*ny*nz where per-voxel off-axis
 *                             distances are written.
 * @param[in]  num_voxels       Device pointer to int[3] containing {nx,ny,nz}.
 * @param[in]  corner           Device pointer to float[3] world-space corner
 *                             coordinates (minimum x,y,z).
 * @param[in]  resolution       Device pointer to float[3] voxel sizes (dx,dy,dz).
 * @param[in]  source_position  Device pointer to float[3] source position.
 * @param[in]  source_v_x       Device pointer to float[3] source local x axis.
 * @param[in]  source_v_y       Device pointer to float[3] source local y axis
 *                             (plane normal used for projection).
 * @param[in]  source_v_z       Device pointer to float[3] source local z axis.
 *
 * @note The kernel uses row-major flattening and expects the host to launch
 *       a grid/block configuration that covers all voxels.
 */
__global__ void oad(float* oad_grid,
    int* num_voxels,
    float* corner,
    float* resolution,
    float* source_position,
    float* source_v_x,
    float* source_v_y,
    float* source_v_z)
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
        float3 source_v_x_f3 = make_float3(source_v_x[0], source_v_x[1], source_v_x[2]);
        float3 source_v_y_f3 = make_float3(source_v_y[0], source_v_y[1], source_v_y[2]);
        float3 source_v_z_f3 = make_float3(source_v_z[0], source_v_z[1], source_v_z[2]);
        float3 position_f3 = make_float3(0.0f, 0.0f, 0.0f);
        float3 distance_f3 = make_float3(0.0f, 0.0f, 0.0f);
        float3 pos_plane_f3 = make_float3(0.0f, 0.0f, 0.0f);
        float3 pos_source_f3 = make_float3(0.0f, 0.0f, 0.0f);

        // Get voxel position
        position_f3.x = corner_f3.x + resolution_f3.x * (x + 0.5);
        position_f3.y = corner_f3.y + resolution_f3.y * (y + 0.5);
        position_f3.z = corner_f3.z + resolution_f3.z * (z + 0.5);

        // Determine distance/direction to source
        distance_f3.x = source_position_f3.x - position_f3.x;
        distance_f3.y = source_position_f3.y - position_f3.y;
        distance_f3.z = source_position_f3.z - position_f3.z;

        // Project position to iso plane
        line_plane_collision_device(&pos_plane_f3, source_position_f3, distance_f3, source_v_y_f3, 1e-6f);

        // Convert to source coords
        pos_source_f3.x = dot3_device(source_v_x_f3, pos_plane_f3);
        pos_source_f3.y = dot3_device(source_v_y_f3, pos_plane_f3);
        pos_source_f3.z = dot3_device(source_v_z_f3, pos_plane_f3);
        int idx = x + y * nx + z * nx * ny;
        oad_grid[idx] = sqrt(pos_source_f3.x * pos_source_f3.x + pos_source_f3.z * pos_source_f3.z);
    }
}

/**
 * @brief Python wrapper that maps NumPy buffers to device memory, launches
 *        the ``oad`` kernel and writes per-voxel off-axis distances back
 *        into the provided output array.
 *
 * This host function is exposed to Python via pybind11 as ``oad``. It:
 *  - obtains raw pointers from the provided NumPy buffers,
 *  - allocates temporary device memory and copies inputs to the GPU,
 *  - launches the CUDA kernel ``oad`` to compute off-axis distances,
 *  - copies the result back into ``oad_grid`` (in-place) and frees device
 *    memory.
 *
 * @note All NumPy arrays passed to this function must be C-contiguous and
 *       have the correct dtypes. The function performs synchronous host-to-
 *       device and device-to-host copies on the default stream.
 *
 * @param[in,out] oad_grid          numpy.ndarray (float32, shape=(nx,ny,nz))
 *      Preallocated output buffer that will contain per-voxel OAD values on
 *      return. Treated as a flattened array of length nx*ny*nz.
 * @param[in] num_voxels            numpy.ndarray (int32, shape=(3,)) holding [nx,ny,nz].
 * @param[in] corner                numpy.ndarray (float32, shape=(3,)) grid corner (min x,y,z).
 * @param[in] resolution            numpy.ndarray (float32, shape=(3,)) voxel sizes (dx,dy,dz).
 * @param[in] source_position       numpy.ndarray (float32, shape=(3,)) world-space source position.
 * @param[in] source_v_x            numpy.ndarray (float32, shape=(3,)) source local x axis.
 * @param[in] source_v_y            numpy.ndarray (float32, shape=(3,)) source local y axis (plane normal).
 * @param[in] source_v_z            numpy.ndarray (float32, shape=(3,)) source local z axis.
 *
 * @par Example (Python)
 * @code{.py}
 * import numpy as np
 * import conehead_gpu as gpu
 * nx,ny,nz = 201,201,201
 * oad = np.zeros((nx,ny,nz), dtype=np.float32)
 * num_voxels = np.array([nx,ny,nz], dtype=np.int32)
 * corner = np.array([-20.1, 0.0, -20.1], dtype=np.float32)
 * resolution = np.array([0.2,0.2,0.2], dtype=np.float32)
 * source_position = np.array([0.0, -100.0, 0.0], dtype=np.float32)
 * source_v_x = np.array([1.0,0.0,0.0], dtype=np.float32)
 * source_v_y = np.array([0.0,1.0,0.0], dtype=np.float32)
 * source_v_z = np.array([0.0,0.0,1.0], dtype=np.float32)
 * gpu.oad(oad, num_voxels, corner, resolution, source_position, source_v_x, source_v_y, source_v_z)
 * @endcode
 */
void map_oad(pybind11::array_t<float> oad_grid, pybind11::array_t<int> num_voxels,
    pybind11::array_t<float> corner, pybind11::array_t<float> resolution,
    pybind11::array_t<float> source_position, pybind11::array_t<float> source_v_x,
    pybind11::array_t<float> source_v_y, pybind11::array_t<float> source_v_z)
{
    pybind11::buffer_info oad_info = oad_grid.request();
    pybind11::buffer_info num_voxels_info = num_voxels.request();
    pybind11::buffer_info corner_info = corner.request();
    pybind11::buffer_info resolution_info = resolution.request();
    pybind11::buffer_info source_position_info = source_position.request();
    pybind11::buffer_info source_v_x_info = source_v_x.request();
    pybind11::buffer_info source_v_y_info = source_v_y.request();
    pybind11::buffer_info source_v_z_info = source_v_z.request();

    float* oad_ptr = reinterpret_cast<float*>(oad_info.ptr);
    int* num_voxels_ptr = reinterpret_cast<int*>(num_voxels_info.ptr);
    float* corner_ptr = reinterpret_cast<float*>(corner_info.ptr);
    float* resolution_ptr = reinterpret_cast<float*>(resolution_info.ptr);
    float* source_position_ptr = reinterpret_cast<float*>(source_position_info.ptr);
    float* source_v_x_ptr = reinterpret_cast<float*>(source_v_x_info.ptr);
    float* source_v_y_ptr = reinterpret_cast<float*>(source_v_y_info.ptr);
    float* source_v_z_ptr = reinterpret_cast<float*>(source_v_z_info.ptr);

    // Allocate device memory and copy inputs
    float *d_oad_grid, *d_corner, *d_resolution, *d_source_position, *d_source_v_x, *d_source_v_y, *d_source_v_z;
    int* d_num_voxels;
    cudaMalloc(&d_oad_grid, oad_info.size * sizeof(float));
    cudaMalloc(&d_num_voxels, num_voxels_info.size * sizeof(int));
    cudaMalloc(&d_corner, corner_info.size * sizeof(float));
    cudaMalloc(&d_resolution, resolution_info.size * sizeof(float));
    cudaMalloc(&d_source_position, source_position_info.size * sizeof(float));
    cudaMalloc(&d_source_v_x, source_v_x_info.size * sizeof(float));
    cudaMalloc(&d_source_v_y, source_v_y_info.size * sizeof(float));
    cudaMalloc(&d_source_v_z, source_v_z_info.size * sizeof(float));
    cudaMemcpy(d_oad_grid, oad_ptr, oad_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_voxels, num_voxels_ptr, num_voxels_info.size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_corner, corner_ptr, corner_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_resolution, resolution_ptr, resolution_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_position, source_position_ptr, source_position_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_v_x, source_v_x_ptr, source_v_x_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_v_y, source_v_y_ptr, source_v_y_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_v_z, source_v_z_ptr, source_v_z_info.size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(16, 4, 4);
    dim3 dimGrid((num_voxels_ptr[0] + dimBlock.x - 1) / dimBlock.x,
        (num_voxels_ptr[1] + dimBlock.y - 1) / dimBlock.y,
        (num_voxels_ptr[2] + dimBlock.z - 1) / dimBlock.z);

    oad<<<dimGrid, dimBlock>>>(d_oad_grid, d_num_voxels, d_corner,
        d_resolution, d_source_position, d_source_v_x, d_source_v_y, d_source_v_z);

    // Copy result back to host
    cudaMemcpy(oad_ptr, d_oad_grid,
        oad_info.size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_oad_grid);
    cudaFree(d_num_voxels);
    cudaFree(d_corner);
    cudaFree(d_resolution);
    cudaFree(d_source_position);
    cudaFree(d_source_v_x);
    cudaFree(d_source_v_y);
    cudaFree(d_source_v_z);
}
