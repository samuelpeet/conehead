#include "device_functions.cuh"
#include "texture_utils.cuh"
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

/**
 * @brief CUDA kernel that computes spectrally-weighted TERMA per voxel.
 *
 * This __global__ kernel evaluates a spectral sum at each voxel using the
 * local incident fluence, material density and per-energy attenuation/deposition
 * parameters. For each energy bin, the kernel traces a ray back to the source,
 * accumulating optical depth (tau) through the density grid, then applies
 * attenuation and energy deposition.
 *
 * @param[out] terma_grid           Device pointer to flattened float array (nx*ny*nz) where TERMA is written.
 * @param[in]  fluence_grid         Device pointer to flattened fluence grid (nx*ny*nz).
 * @param[in]  d_geo_grid           Device pointer to flattened geometric distances (nx*ny*nz).
 * @param[in]  d_eff_grid           Device pointer to flattened radiological/effective depth grid (nx*ny*nz).
 * @param[in]  density_grid         Device pointer to flattened density grid (nx*ny*nz).
 * @param[in]  num_energies         Number of spectral energy bins.
 * @param[in]  energies             Device pointer to per-bin energy values (float[num_energies]).
 * @param[in]  energy_weights       Device pointer to per-bin normalized weights (float[num_energies]).
 * @param[in]  mu_tot           Device pointer to per-bin linear attenuation coefficients (float[num_energies]).
 * @param[in]  mu_en            Device pointer to per-bin energy absorption coefficients (float[num_energies]).
 * @param[in]  oad_grid             Device pointer to off-axis-distance grid (nx*ny*nz).
 * @param[in]  off_axis_softening_fs_interp Device pointer to LUT for off-axis softening.
 * @param[in]  off_axis_softening_dx Grid spacing for off-axis softening LUT.
 * @param[in]  off_axis_softening_oad_max Maximum off-axis distance for softening LUT.
 * @param[in]  num_voxels           Device pointer to int[3] = {nx,ny,nz}.
 * @param[in]  corner               Device pointer to float[3] grid corner position {x,y,z}.
 * @param[in]  resolution           Device pointer to float[3] voxel spacing {dx,dy,dz}.
 * @param[in]  source_position      Device pointer to float[3] source position {x,y,z}.
 * @param[in]  source_sad           Source-to-axis distance used for no-tilt descaling.
 *
 * @note Each thread traces a ray back to the source, accumulating attenuation
 *       through density and material-dependent cross-sections, then applies
 *       spectral weighting and inverse-square descaling.
 */
__global__ void terma_spectral(float* terma_grid,
    float* fluence_grid,
    float* d_geo_grid,
    float* d_eff_grid,
    float* density_grid,
    int num_energies,
    float* energies,
    float* energy_weights,
    float* mu_tot,
    float* mu_en,
    float* oad_grid,
    float* off_axis_softening_fs_interp,
    float off_axis_softening_dx,
    float off_axis_softening_oad_max,
    int* num_voxels,
    float* corner,
    float* resolution,
    float* source_position,
    float source_sad)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = num_voxels[0];
    int ny = num_voxels[1];
    int nz = num_voxels[2];

    if (x < nx && y < ny && z < nz) {
        int idx = x + y * nx + z * nx * ny;

        const int n_en = 12;
        float taus[n_en];

        float oad = oad_grid[idx];
        int ix = (int)(oad / off_axis_softening_dx);
        ix = min(ix, (int)(off_axis_softening_oad_max / off_axis_softening_dx) - 1);
        float oas = off_axis_softening_fs_interp[ix];
        
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
        
        for (int i = 0; i < num_energies; i++) {
            float tau = 0;
            float3 ray_pos = position_f3;  // Use a copy for ray tracing
            for (int s = 0; s < steps; s++) {
                 // Move a step toward source
                ray_pos.x += ray_direction_f3.x * ds;
                ray_pos.y += ray_direction_f3.y * ds;
                ray_pos.z += ray_direction_f3.z * ds;
                // Map position -> continuous indices (x fastest)
                float fx = (ray_pos.x - corner_f3.x) / resolution_f3.x;
                float fy = (ray_pos.y - corner_f3.y) / resolution_f3.y;
                float fz = (ray_pos.z - corner_f3.z) / resolution_f3.z;
                int ix = __float2int_rd(fx);
                int iy = __float2int_rd(fy);
                int iz = __float2int_rd(fz);
                if ((unsigned)ix >= nx || (unsigned)iy >= ny || (unsigned)iz >= nz) {
                    break; // Ray left grid
                }
                int idx2 = ix + iy * nx + iz * nx * ny;
                
                float density = density_grid[idx2];

                tau += mu_tot[i] * density * ds;
            }
            taus[i] = tau;
        }

        float terma = 0;
        float fluence = fluence_grid[idx];
        for (int i = 0; i < num_energies; i++) {
            // float d_eff = d_eff_grid[idx] + oas;
            terma += energy_weights[i] * fluence * energies[i] * expf(-(taus[i] + oas)) * mu_en[i];
        }
        float d_geo = d_geo_grid[idx];
        float no_tilt_descaling = (d_geo / source_sad) * (d_geo / source_sad);
        terma_grid[idx] = terma * no_tilt_descaling;
    }
}

/**
 * @brief Python-visible host wrapper that prepares inputs and launches the `terma_spectral` kernel.
 *
 * This function is exposed to Python via pybind11. It accepts NumPy arrays
 * for TERMA/fluence/d_geo/d_eff/density, spectral arrays, material grid,
 * and off-axis LUTs; it copies the inputs to device memory, launches the
 * `terma_spectral` kernel and copies the computed TERMA grid back into the provided
 * output array.
 *
 * All NumPy arrays are expected to be C-contiguous and dtype float32
 * (except `num_voxels` and `material_grid` which should be int32). The `terma_grid` array is
 * overwritten in-place.
 *
 * @param terma_grid        NumPy array (float32) flattened output buffer for nx*ny*nz elements.
 * @param fluence_grid      NumPy array (float32) flattened incident fluence grid.
 * @param d_geo_grid        NumPy array (float32) flattened geometric distances.
 * @param d_eff_grid        NumPy array (float32) flattened radiological depths.
 * @param density_grid      NumPy array (float32) flattened density grid.
 * @param num_energies      Integer: number of spectral bins.
 * @param energies          NumPy array (float32) per-bin energies.
 * @param energy_weights    NumPy array (float32) per-bin weights (should sum to 1).
 * @param mu_tot            NumPy array (float32) per-bin linear attenuation coefficients (num_energies).
 * @param mu_en             NumPy array (float32) per-bin energy absorption coefficients (num_energies).
 * @param oad_grid          NumPy array (float32) flattened off-axis distances.
 * @param off_axis_softening_fs_interp NumPy array (float32) LUT for off-axis softening.
 * @param off_axis_softening_dx Float: spacing for the off-axis LUT.
 * @param off_axis_softening_oad_max Float: maximum off-axis distance for softening LUT.
 * @param num_voxels        NumPy array (int32, shape=(3,)) holding [nx,ny,nz].
 * @param corner            NumPy array (float32, shape=(3,)) grid corner {x,y,z}.
 * @param resolution        NumPy array (float32, shape=(3,)) voxel spacing {dx,dy,dz}.
 * @param source_position   NumPy array (float32, shape=(3,)) source position {x,y,z}.
 * @param source_sad        Float: source-to-axis distance used for descaling.
 *
 * @note This wrapper performs synchronous host↔device copies on the default
 *       stream. For repeated calls with the same large arrays consider
 *       exposing persistent device buffers or texture handles to avoid
 *       allocation and copy overhead.
 */
void map_terma_spectral(
    pybind11::array_t<float> terma_grid,
    pybind11::array_t<float> fluence_grid,
    pybind11::array_t<float> d_geo_grid,
    pybind11::array_t<float> d_eff_grid,
    pybind11::array_t<float> density_grid,
    int num_energies,
    pybind11::array_t<float> energies,
    pybind11::array_t<float> energy_weights,
    pybind11::array_t<float> mu_tot,
    pybind11::array_t<float> mu_en,
    pybind11::array_t<float> oad_grid,
    pybind11::array_t<float> off_axis_softening_fs_interp,
    float off_axis_softening_dx,
    float off_axis_softening_oad_max,
    pybind11::array_t<int> num_voxels,
    pybind11::array_t<float> corner,
    pybind11::array_t<float> resolution,
    pybind11::array_t<float> source_position,
    float source_sad)
{
    pybind11::buffer_info terma_grid_info = terma_grid.request();
    pybind11::buffer_info fluence_grid_info = fluence_grid.request();
    pybind11::buffer_info d_geo_grid_info = d_geo_grid.request();
    pybind11::buffer_info d_eff_grid_info = d_eff_grid.request();
    pybind11::buffer_info density_grid_info = density_grid.request();
    pybind11::buffer_info num_voxels_info = num_voxels.request();
    pybind11::buffer_info corner_info = corner.request();
    pybind11::buffer_info resolution_info = resolution.request();
    pybind11::buffer_info source_position_info = source_position.request();
    pybind11::buffer_info energies_info = energies.request();
    pybind11::buffer_info energy_weights_info = energy_weights.request();
    pybind11::buffer_info mu_tot_info = mu_tot.request();
    pybind11::buffer_info mu_en_info = mu_en.request();
    pybind11::buffer_info oad_grid_info = oad_grid.request();
    pybind11::buffer_info off_axis_softening_fs_interp_info = off_axis_softening_fs_interp.request();

    float* terma_grid_ptr = reinterpret_cast<float*>(terma_grid_info.ptr);
    float* fluence_grid_ptr = reinterpret_cast<float*>(fluence_grid_info.ptr);
    float* d_geo_grid_ptr = reinterpret_cast<float*>(d_geo_grid_info.ptr);
    float* d_eff_grid_ptr = reinterpret_cast<float*>(d_eff_grid_info.ptr);
    float* density_grid_ptr = reinterpret_cast<float*>(density_grid_info.ptr);
    int* num_voxels_ptr = reinterpret_cast<int*>(num_voxels_info.ptr);
    float* corner_ptr = reinterpret_cast<float*>(corner_info.ptr);
    float* resolution_ptr = reinterpret_cast<float*>(resolution_info.ptr);
    float* source_position_ptr = reinterpret_cast<float*>(source_position_info.ptr);
    float* energies_ptr = reinterpret_cast<float*>(energies_info.ptr);
    float* energy_weights_ptr = reinterpret_cast<float*>(energy_weights_info.ptr);
    float* mu_tot_ptr = reinterpret_cast<float*>(mu_tot_info.ptr);
    float* mu_en_ptr = reinterpret_cast<float*>(mu_en_info.ptr);
    float* oad_grid_ptr = reinterpret_cast<float*>(oad_grid_info.ptr);
    float* off_axis_softening_fs_interp_ptr = reinterpret_cast<float*>(off_axis_softening_fs_interp_info.ptr);

    // Allocate device memory and copy inputs
    float *d_terma_grid, *d_fluence_grid, *d_d_geo_grid, *d_d_eff_grid, *d_density_grid;
    float *d_energies, *d_energy_weights, *d_mu_tot, *d_mu_en, *d_oad_grid, *d_off_axis_softening_fs_interp;
    float *d_corner, *d_resolution, *d_source_position;
    int *d_num_voxels;
    
    cudaMalloc(&d_terma_grid, terma_grid_info.size * sizeof(float));
    cudaMalloc(&d_fluence_grid, fluence_grid_info.size * sizeof(float));
    cudaMalloc(&d_d_geo_grid, d_geo_grid_info.size * sizeof(float));
    cudaMalloc(&d_d_eff_grid, d_eff_grid_info.size * sizeof(float));
    cudaMalloc(&d_density_grid, density_grid_info.size * sizeof(float));
    cudaMalloc(&d_num_voxels, num_voxels_info.size * sizeof(int));
    cudaMalloc(&d_corner, corner_info.size * sizeof(float));
    cudaMalloc(&d_resolution, resolution_info.size * sizeof(float));
    cudaMalloc(&d_source_position, source_position_info.size * sizeof(float));
    cudaMalloc(&d_energies, energies_info.size * sizeof(float));
    cudaMalloc(&d_energy_weights, energy_weights_info.size * sizeof(float));
    cudaMalloc(&d_mu_tot, mu_tot_info.size * sizeof(float));
    cudaMalloc(&d_mu_en, mu_en_info.size * sizeof(float));
    cudaMalloc(&d_oad_grid, oad_grid_info.size * sizeof(float));
    cudaMalloc(&d_off_axis_softening_fs_interp, off_axis_softening_fs_interp_info.size * sizeof(float));
    
    cudaMemcpy(d_terma_grid, terma_grid_ptr, terma_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fluence_grid, fluence_grid_ptr, fluence_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_geo_grid, d_geo_grid_ptr, d_geo_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_eff_grid, d_eff_grid_ptr, d_eff_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_density_grid, density_grid_ptr, density_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_voxels, num_voxels_ptr, num_voxels_info.size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_corner, corner_ptr, corner_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_resolution, resolution_ptr, resolution_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_position, source_position_ptr, source_position_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energies, energies_ptr, energies_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energy_weights, energy_weights_ptr, energy_weights_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu_tot, mu_tot_ptr, mu_tot_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu_en, mu_en_ptr, mu_en_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oad_grid, oad_grid_ptr, oad_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_off_axis_softening_fs_interp, off_axis_softening_fs_interp_ptr, off_axis_softening_fs_interp_info.size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(16, 4, 4);
    dim3 dimGrid((num_voxels_ptr[0] + dimBlock.x - 1) / dimBlock.x,
        (num_voxels_ptr[1] + dimBlock.y - 1) / dimBlock.y,
        (num_voxels_ptr[2] + dimBlock.z - 1) / dimBlock.z);

    terma_spectral<<<dimGrid, dimBlock>>>(
        d_terma_grid,
        d_fluence_grid,
        d_d_geo_grid,
        d_d_eff_grid,
        d_density_grid,
        num_energies,
        d_energies,
        d_energy_weights,
        d_mu_tot,
        d_mu_en,
        d_oad_grid,
        d_off_axis_softening_fs_interp,
        off_axis_softening_dx,
        off_axis_softening_oad_max,
        d_num_voxels,
        d_corner,
        d_resolution,
        d_source_position,
        source_sad);

    // Copy result back to host
    cudaMemcpy(terma_grid_ptr, d_terma_grid,
        terma_grid_info.size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_terma_grid);
    cudaFree(d_fluence_grid);
    cudaFree(d_d_geo_grid);
    cudaFree(d_d_eff_grid);
    cudaFree(d_density_grid);
    cudaFree(d_num_voxels);
    cudaFree(d_corner);
    cudaFree(d_resolution);
    cudaFree(d_source_position);
    cudaFree(d_energies);
    cudaFree(d_energy_weights);
    cudaFree(d_mu_tot);
    cudaFree(d_mu_en);
    cudaFree(d_oad_grid);
    cudaFree(d_off_axis_softening_fs_interp);
}
