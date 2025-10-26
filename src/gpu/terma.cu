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
 * local incident fluence, radiological depth (d_eff), material density and
 * per-energy attenuation/deposition parameters. The per-voxel TERMA is
 * computed as::
 *
 *     terma = sum_i energy_weights[i] * fluence * exp(-mu_w[i] * d_eff) * mu_w[i] * rho * energy[i]
 *
 * After the spectral sum the kernel applies a "no-tilt" inverse-square
 * descaling so that dose convolution can later reapply geometric scaling.
 *
 * @param[out] terma_grid       Device pointer to flattened float array (nx*ny*nz) where TERMA is written.
 * @param[in]  fluence_grid     Device pointer to flattened fluence grid (nx*ny*nz).
 * @param[in]  d_geo_grid       Device pointer to flattened geometric distances (nx*ny*nz).
 * @param[in]  d_eff_grid       Device pointer to flattened radiological/effective depth grid (nx*ny*nz).
 * @param[in]  num_voxels       Device pointer to int[3] = {nx,ny,nz}.
 * @param[in]  num_energies     Number of spectral energy bins (length of energies/weights arrays).
 * @param[in]  energies         Device pointer to per-bin energy values (float[num_energies]).
 * @param[in]  energy_weights   Device pointer to per-bin normalized weights (float[num_energies]).
 * @param[in]  mu_w             Device pointer to per-bin linear attenuation coefficients (float[num_energies]).
 * @param[in]  source_sad       Source-to-axis distance used for no-tilt descaling.
 * @param[in]  oad_grid         Device pointer to off-axis-distance grid (nx*ny*nz) for optional off-axis softening (unused here).
 * @param[in]  off_axis_softening_fs_interp Device pointer to LUT for off-axis softening (optional/unused).
 * @param[in]  off_axis_softening_dx Grid spacing for off-axis softening LUT (optional/unused).
 *
 * @note Off-axis softening is not applied in the current implementation; the
 *       commented code shows an approach that would modify per-bin weights
 *       using a LUT indexed by the off-axis distance.
 */
__global__ void terma(float* terma_grid,
    float* fluence_grid,
    float* d_geo_grid,
    float* d_eff_grid,
    int* num_voxels,
    int num_energies,
    float* energies,
    float* energy_weights,
    float* mu_w,
    float source_sad,
    float* oad_grid,
    float* off_axis_softening_fs_interp,
    float off_axis_softening_dx)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = num_voxels[0];
    int ny = num_voxels[1];
    int nz = num_voxels[2];

    if (x < nx && y < ny && z < nz) {
        int idx = x + y * nx + z * nx * ny;

        // float oad = oad_grid[idx];
        // int ix = (int)(oad / off_axis_softening_dx);
        // float oas = off_axis_softening_fs_interp[ix];
        //
        // float new_energy_weights[12]; // Assuming a maximum of 12 energy bins
        // for (int i = 0; i < num_energies; i++)
        // {
        //     new_energy_weights[i] = energy_weights[i] * exp(mu_w[i] * (oas + d_eff_grid[idx]));
        // }
        // float sum_weights = 0;
        // for (int i = 0; i < num_energies; i++)
        // {
        //     sum_weights += new_energy_weights[i];
        // }
        // for (int i = 0; i < num_energies; i++)
        // {
        //     new_energy_weights[i] /= sum_weights;
        // }

        float terma = 0;
        for (int i = 0; i < num_energies; i++) {
            float fluence = fluence_grid[idx];
            float d_eff = d_eff_grid[idx];
            terma += energy_weights[i] * fluence * expf(-mu_w[i] * d_eff) * mu_w[i] * energies[i];
        }
        float d_geo = d_geo_grid[idx];
        float no_tilt_descaling = (d_geo / source_sad) * (d_geo / source_sad);
        terma_grid[idx] = terma * no_tilt_descaling;
    }
}

/**
 * @brief Python-visible host wrapper that prepares inputs and launches the `terma` kernel.
 *
 * This function is exposed to Python via pybind11. It accepts NumPy arrays
 * for TERMA/fluence/d_geo/d_eff/density, spectral arrays and optional
 * off-axis LUTs; it copies the inputs to device memory, launches the
 * `terma` kernel and copies the computed TERMA grid back into the provided
 * output array.
 *
 * All NumPy arrays are expected to be C-contiguous and dtype float32
 * (except `num_voxels` which should be int32). The `terma_grid` array is
 * overwritten in-place.
 *
 * @param terma_grid        NumPy array (float32) flattened output buffer for nx*ny*nz elements.
 * @param fluence_grid      NumPy array (float32) flattened incident fluence grid.
 * @param d_geo_grid        NumPy array (float32) flattened geometric distances.
 * @param d_eff_grid        NumPy array (float32) flattened radiological depths.
 * @param num_voxels        NumPy array (int32, shape=(3,)) holding [nx,ny,nz].
 * @param num_energies      Integer: number of spectral bins.
 * @param energies          NumPy array (float32) per-bin energies.
 * @param energy_weights    NumPy array (float32) per-bin weights (should sum to 1).
 * @param mu_w              NumPy array (float32) per-bin linear attenuation coefficients.
 * @param source_sad        Float: source-to-axis distance used for descaling.
 * @param oad_grid          NumPy array (float32) flattened off-axis distances (optional, can be zeros).
 * @param off_axis_softening_fs_interp NumPy array (float32) LUT for off-axis softening (optional).
 * @param off_axis_softening_dx Float: spacing for the off-axis LUT (optional).
 *
 * @note This wrapper performs synchronous host↔device copies on the default
 *       stream. For repeated calls with the same large arrays consider
 *       exposing persistent device buffers or texture handles to avoid
 *       allocation and copy overhead.
 */
void map_terma(
    pybind11::array_t<float> terma_grid,
    pybind11::array_t<float> fluence_grid,
    pybind11::array_t<float> d_geo_grid,
    pybind11::array_t<float> d_eff_grid,
    pybind11::array_t<int> num_voxels,
    int num_energies,
    pybind11::array_t<float> energies,
    pybind11::array_t<float> energy_weights,
    pybind11::array_t<float> mu_w,
    float source_sad,
    pybind11::array_t<float> oad_grid,
    pybind11::array_t<float> off_axis_softening_fs_interp,
    float off_axis_softening_dx)
{
    pybind11::buffer_info terma_grid_info = terma_grid.request();
    pybind11::buffer_info fluence_grid_info = fluence_grid.request();
    pybind11::buffer_info d_geo_grid_info = d_geo_grid.request();
    pybind11::buffer_info d_eff_grid_info = d_eff_grid.request();
    pybind11::buffer_info num_voxels_info = num_voxels.request();
    pybind11::buffer_info energies_info = energies.request();
    pybind11::buffer_info energy_weights_info = energy_weights.request();
    pybind11::buffer_info mu_w_info = mu_w.request();
    pybind11::buffer_info oad_grid_info = oad_grid.request();
    pybind11::buffer_info off_axis_softening_fs_interp_info = off_axis_softening_fs_interp.request();

    float* terma_grid_ptr = reinterpret_cast<float*>(terma_grid_info.ptr);
    float* fluence_grid_ptr = reinterpret_cast<float*>(fluence_grid_info.ptr);
    float* d_geo_grid_ptr = reinterpret_cast<float*>(d_geo_grid_info.ptr);
    float* d_eff_grid_ptr = reinterpret_cast<float*>(d_eff_grid_info.ptr);
    int* num_voxels_ptr = reinterpret_cast<int*>(num_voxels_info.ptr);
    float* energies_ptr = reinterpret_cast<float*>(energies_info.ptr);
    float* energy_weights_ptr = reinterpret_cast<float*>(energy_weights_info.ptr);
    float* mu_w_ptr = reinterpret_cast<float*>(mu_w_info.ptr);
    float* oad_grid_ptr = reinterpret_cast<float*>(oad_grid_info.ptr);
    float* off_axis_softening_fs_interp_ptr = reinterpret_cast<float*>(off_axis_softening_fs_interp_info.ptr);

    // Allocate device memory and copy inputs
    float *d_terma_grid, *d_fluence_grid, *d_d_geo_grid, *d_d_eff_grid;
    float *d_energies, *d_energy_weights, *d_mu_w, *d_oad_grid, *d_off_axis_softening_fs_interp;
    int* d_num_voxels;
    cudaMalloc(&d_terma_grid, terma_grid_info.size * sizeof(float));
    cudaMalloc(&d_fluence_grid, fluence_grid_info.size * sizeof(float));
    cudaMalloc(&d_d_geo_grid, d_geo_grid_info.size * sizeof(float));
    cudaMalloc(&d_d_eff_grid, d_eff_grid_info.size * sizeof(float));
    cudaMalloc(&d_num_voxels, num_voxels_info.size * sizeof(int));
    cudaMalloc(&d_energies, energies_info.size * sizeof(float));
    cudaMalloc(&d_energy_weights, energy_weights_info.size * sizeof(float));
    cudaMalloc(&d_mu_w, mu_w_info.size * sizeof(float));
    cudaMalloc(&d_oad_grid, oad_grid_info.size * sizeof(float));
    cudaMalloc(&d_off_axis_softening_fs_interp, off_axis_softening_fs_interp_info.size * sizeof(float));
    cudaMemcpy(d_terma_grid, terma_grid_ptr, terma_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fluence_grid, fluence_grid_ptr, fluence_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_geo_grid, d_geo_grid_ptr, d_geo_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_eff_grid, d_eff_grid_ptr, d_eff_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_voxels, num_voxels_ptr, num_voxels_info.size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energies, energies_ptr, energies_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energy_weights, energy_weights_ptr, energy_weights_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu_w, mu_w_ptr, mu_w_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oad_grid, oad_grid_ptr, oad_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_off_axis_softening_fs_interp, off_axis_softening_fs_interp_ptr, off_axis_softening_fs_interp_info.size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(16, 4, 4);
    dim3 dimGrid((num_voxels_ptr[0] + dimBlock.x - 1) / dimBlock.x,
        (num_voxels_ptr[1] + dimBlock.y - 1) / dimBlock.y,
        (num_voxels_ptr[2] + dimBlock.z - 1) / dimBlock.z);

    // Pass texture objects for the fluence maps and d_geo
    terma<<<dimGrid, dimBlock>>>(
        d_terma_grid,
        d_fluence_grid,
        d_d_geo_grid,
        d_d_eff_grid,
        d_num_voxels,
        num_energies,
        d_energies,
        d_energy_weights,
        d_mu_w,
        source_sad,
        d_oad_grid,
        d_off_axis_softening_fs_interp,
        off_axis_softening_dx);

    // Copy result back to host
    cudaMemcpy(terma_grid_ptr, d_terma_grid,
        terma_grid_info.size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_terma_grid);
    cudaFree(d_fluence_grid);
    cudaFree(d_d_geo_grid);
    cudaFree(d_d_eff_grid);
    cudaFree(d_num_voxels);
    cudaFree(d_energies);
    cudaFree(d_energy_weights);
    cudaFree(d_mu_w);
    cudaFree(d_oad_grid);
    cudaFree(d_off_axis_softening_fs_interp);
}
