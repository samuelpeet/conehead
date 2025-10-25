#include "texture_utils.cuh"
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

/**
 * @brief CUDA kernel that computes radiological (effective) depth toward a point source.
 *
 * This __global__ kernel assigns one CUDA thread per voxel. Each thread
 * marches from the centre of its voxel toward the provided ``source_position``
 * in small steps of length ``ds = 0.25 * min(dx,dy,dz)`` (where dx/dy/dz are
 * the voxel resolutions). At each step the kernel samples the provided
 * ``density_grid`` and accumulates ``density * ds`` into a running total. The
 * final accumulated radiological depth is written to ``d_eff_grid`` at the
 * flattened index ``idx = x + y*nx + z*nx*ny``.
 *
 * @param[in,out] d_eff_grid     Device pointer to a flattened float array of length nx*ny*nz
 *                           where per-voxel effective depths are written.
 * @param[in]  num_voxels     Device pointer to an int[3] containing {nx, ny, nz}.
 * @param[in]  corner         Device pointer to float[3] world-space corner coordinates (min x,y,z).
 * @param[in]  resolution     Device pointer to float[3] voxel sizes (dx, dy, dz).
 * @param[in]  density_grid   Device pointer to flattened density grid (same layout as d_eff_grid).
 * @param[in]  source_position Device pointer to float[3] source position in world coordinates.
 *
 * @note The kernel uses a fixed step length derived from the voxel size to
 *       provide a balance between accuracy and performance. For higher
 *       accuracy you can reduce the step fraction or add adaptive stepping.
 */
__global__ void d_eff(float* d_eff_grid,
    int* num_voxels,
    float* corner,
    float* resolution,
    cudaTextureObject_t density_tex,
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

        // Determine direction to source
        ray_direction_f3.x = source_position_f3.x - position_f3.x;
        ray_direction_f3.y = source_position_f3.y - position_f3.y;
        ray_direction_f3.z = source_position_f3.z - position_f3.z;
        float mag = sqrt(ray_direction_f3.x * ray_direction_f3.x + ray_direction_f3.y * ray_direction_f3.y + ray_direction_f3.z * ray_direction_f3.z);
        ray_direction_f3.x /= mag;
        ray_direction_f3.y /= mag;
        ray_direction_f3.z /= mag;

        // Compute steplength and number of steps
        float ds = 0.25 * fmin(fmin(resolution_f3.x, resolution_f3.y), resolution_f3.z);
        float total_distance = mag;
        int steps = (int)(total_distance / ds);
        float acc = 0;

        for (int i = 0; i < steps; i++) {
            // Move a step toward source
            position_f3.x += ray_direction_f3.x * ds;
            position_f3.y += ray_direction_f3.y * ds;
            position_f3.z += ray_direction_f3.z * ds;
            // Map position -> continuous indices (x fastest)
            float fx = (position_f3.x - corner_f3.x) / resolution_f3.x;
            float fy = (position_f3.y - corner_f3.y) / resolution_f3.y;
            float fz = (position_f3.z - corner_f3.z) / resolution_f3.z;

            // Bounds check: if sample point is outside grid, stop marching
            if (fx < 0.0f || fx >= (float)nx || fy < 0.0f || fy >= (float)ny || fz < 0.0f || fz >= (float)nz) {
                break; // Ray left grid
            }

            // Sample density via 3D texture. Use texel-centred coordinates (add 0.5)
            float sample = tex3D<float>(density_tex, fx + 0.5f, fy + 0.5f, fz + 0.5f);
            acc += sample * ds;
        }
        // Store result
        int idx = x + y * nx + z * nx * ny;
        d_eff_grid[idx] = acc;
    }
}

/**
 * @brief Python wrapper that maps NumPy buffers to device memory, launches
 *        the ``d_eff`` kernel and writes per-voxel radiological depth values
 *        back into the provided output array.
 *
 * This host function is exposed to Python via pybind11 as ``d_eff``. It:
 *  - extracts raw pointers from provided NumPy arrays (expects C-contiguous
 *    arrays of the correct dtype),
 *  - allocates temporary device buffers and copies inputs to the GPU,
 *  - launches the CUDA kernel ``d_eff`` to perform ray-marching integration,
 *  - copies results back into ``d_eff_grid`` (in-place) and frees device memory.
 *
 * @param[in,out] d_eff_grid    numpy.ndarray (float32, shape=(nx,ny,nz)) Preallocated output buffer.
 * @param[in]     num_voxels    numpy.ndarray (int32, shape=(3,)) grid dimensions [nx,ny,nz].
 * @param[in]     corner        numpy.ndarray (float32, shape=(3,)) grid corner (min x,y,z).
 * @param[in]     resolution    numpy.ndarray (float32, shape=(3,)) voxel sizes (dx,dy,dz).
 * @param[in]     density_grid  numpy.ndarray (float32, shape=(nx,ny,nz)) flattened density grid.
 * @param[in]     source_position numpy.ndarray (float32, shape=(3,)) world-space source position.
 *
 * @par Threading / Streams
 * The wrapper performs synchronous host-device copies and launches the kernel
 * on the default stream. For large grids you may want to add an asynchronous
 * API that accepts a cudaStream_t and uses pinned memory for overlap.
 *
 * @par Example (Python)
 * @code{.py}
 * import numpy as np
 * import conehead_gpu as gpu
 * nx,ny,nz = 201,201,201
 * d_eff = np.zeros((nx,ny,nz), dtype=np.float32)
 * num_voxels = np.array([nx,ny,nz], dtype=np.int32)
 * corner = np.array([-20.1, 0.0, -20.1], dtype=np.float32)
 * resolution = np.array([0.2,0.2,0.2], dtype=np.float32)
 * density = np.ones((nx,ny,nz), dtype=np.float32) # example homogeneous density
 * source_position = np.array([0.0, -100.0, 0.0], dtype=np.float32)
 * gpu.d_eff(d_eff, num_voxels, corner, resolution, density, source_position)
 * @endcode
 */
void map_d_eff(pybind11::array_t<float> d_eff_grid, pybind11::array_t<int> num_voxels,
    pybind11::array_t<float> corner, pybind11::array_t<float> resolution, pybind11::array_t<float> density_grid,
    pybind11::array_t<float> source_position)
{
    pybind11::buffer_info d_eff_info = d_eff_grid.request();
    pybind11::buffer_info num_voxels_info = num_voxels.request();
    pybind11::buffer_info corner_info = corner.request();
    pybind11::buffer_info resolution_info = resolution.request();
    pybind11::buffer_info density_grid_info = density_grid.request();
    pybind11::buffer_info source_position_info = source_position.request();

    float* d_eff_ptr = reinterpret_cast<float*>(d_eff_info.ptr);
    int* num_voxels_ptr = reinterpret_cast<int*>(num_voxels_info.ptr);
    float* corner_ptr = reinterpret_cast<float*>(corner_info.ptr);
    float* resolution_ptr = reinterpret_cast<float*>(resolution_info.ptr);
    float* density_grid_ptr = reinterpret_cast<float*>(density_grid_info.ptr);
    float* source_position_ptr = reinterpret_cast<float*>(source_position_info.ptr);

    // Allocate device memory and copy inputs
    float *d_d_eff_grid, *d_corner, *d_resolution, *d_source_position;
    int* d_num_voxels;
    cudaMalloc(&d_d_eff_grid, d_eff_info.size * sizeof(float));
    cudaMalloc(&d_num_voxels, num_voxels_info.size * sizeof(int));
    cudaMalloc(&d_corner, corner_info.size * sizeof(float));
    cudaMalloc(&d_resolution, resolution_info.size * sizeof(float));
    cudaMalloc(&d_source_position, source_position_info.size * sizeof(float));
    cudaMemcpy(d_d_eff_grid, d_eff_ptr, d_eff_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_voxels, num_voxels_ptr, num_voxels_info.size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_corner, corner_ptr, corner_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_resolution, resolution_ptr, resolution_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_position, source_position_ptr, source_position_info.size * sizeof(float), cudaMemcpyHostToDevice);

    // Create texture object from host density array (helper in texture_utils.cuh)
    Texture3DHandle texh = create_texture3d_from_ptr(density_grid_ptr, num_voxels_ptr[0], num_voxels_ptr[1], num_voxels_ptr[2], cudaFilterModeLinear);
    cudaTextureObject_t density_tex = texh.tex;

    // Launch kernel
    dim3 dimBlock(16, 4, 4);
    dim3 dimGrid((num_voxels_ptr[0] + dimBlock.x - 1) / dimBlock.x,
        (num_voxels_ptr[1] + dimBlock.y - 1) / dimBlock.y,
        (num_voxels_ptr[2] + dimBlock.z - 1) / dimBlock.z);

    d_eff<<<dimGrid, dimBlock>>>(d_d_eff_grid, d_num_voxels, d_corner,
        d_resolution, density_tex, d_source_position);

    // Copy result back to host
    cudaMemcpy(d_eff_ptr, d_d_eff_grid,
        d_eff_info.size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_d_eff_grid);
    cudaFree(d_num_voxels);
    cudaFree(d_corner);
    cudaFree(d_resolution);
    cudaFree(d_source_position);

    // Destroy texture and free CUDA array (helper)
    destroy_texture3d(texh);
}
