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
 * local incident fluence, radiological depth (d_eff) and per-energy
 * attenuation/deposition parameters. The implementation computes for each
 * voxel:
 *
 *   terma = sum_i energy_weights[i] * Fluence(x,y,z) * exp(-mu_w[i] * d_eff(x,y,z)) * mu_w[i] * energy[i]
 *
 * and then writes a "no-tilt" descaled TERMA:
 *
 *   terma_out = terma * (d_geo(x,y,z) / source_sad)^2
 *
 * where Fluence, d_eff and d_geo are sampled from 3D texture objects.
 *
 * @param[in,out] terma_grid     Device pointer to flattened float array (nx*ny*nz) where TERMA is written.
 * @param[in]  fluence_grid   3D cudaTextureObject_t containing per-voxel fluence values (unnormalized texel coords).
 * @param[in]  d_geo_grid     3D cudaTextureObject_t containing geometric distances (used for no-tilt descaling).
 * @param[in]  d_eff_grid     3D cudaTextureObject_t containing radiological/effective depths used in attenuation.
 * @param[in]  num_voxels     Device pointer to int[3] = {nx,ny,nz}.
 * @param[in]  num_energies   Number of spectral energy bins (length of energies/weights arrays).
 * @param[in]  energies       Device pointer to per-bin energy values (float[num_energies]).
 * @param[in]  energy_weights Device pointer to per-bin weights (float[num_energies]).
 * @param[in]  mu_w           Device pointer to per-bin linear attenuation coefficients (float[num_energies]).
 * @param[in]  source_sad     Source-to-axis distance used for no-tilt descaling.
 * @param[in]  oad_grid       3D cudaTextureObject_t of off-axis distances (present for future off-axis softening).
 * @param[in]  off_axis_softening_fs_interp Device pointer to LUT for off-axis softening (optional, currently unused).
 * @param[in]  off_axis_softening_dx Grid spacing for off-axis softening LUT (optional, currently unused).
 *
 * @note Off-axis softening is not applied in this implementation; commented
 *       code shows planned LUT-based softening that would adjust per-bin
 *       energy_weights prior to summation.
 */
__global__ void terma(float* terma_grid,
    cudaTextureObject_t fluence_grid,
    cudaTextureObject_t d_geo_grid,
    cudaTextureObject_t d_eff_grid,
    int* num_voxels,
    int num_energies,
    float* energies,
    float* energy_weights,
    float* mu_w,
    float source_sad,
    cudaTextureObject_t oad_grid,
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
            float fluence = tex3D<float>(fluence_grid, x, y, z);
            float d_eff = tex3D<float>(d_eff_grid, x, y, z);
            terma += energy_weights[i] * fluence * expf(-mu_w[i] * d_eff) * mu_w[i] * energies[i];
        }
        float d_geo = tex3D<float>(d_geo_grid, x, y, z);
        float no_tilt_descaling = (d_geo / source_sad) * (d_geo / source_sad);
        terma_grid[idx] = terma * no_tilt_descaling;
    }
}

/**
 * @brief Python-visible host wrapper that prepares inputs and launches the
 *        `fluence` kernel.
 *
 * This function is intended to be bound via pybind11. It accepts NumPy
 * arrays for the output grid, two 2D fluence maps, geometry and source
 * parameters, copies necessary data to device memory, creates texture
 * objects for the fluence maps and the precomputed geometric-distance grid,
 * launches the `fluence` kernel and copies the resulting fluence grid back
 * to the provided output array.
 *
 * All array arguments are expected to be contiguous and of dtype float32
 * (except `num_voxels` which should be int32). The `fluence_grid` array is
 * overwritten with the computed values.
 *
 * @param fluence_grid       NumPy array (float32) shaped (nx*ny*nz,) or compatible flattened buffer.
 * @param fluence_map_pri    NumPy 2D array (float32) containing the primary fluence map.
 * @param fluence_map_sec    NumPy 2D array (float32) containing the secondary fluence map.
 * @param num_voxels         NumPy array of 3 ints: {nx, ny, nz}.
 * @param corner             NumPy array float[3] world-space grid corner (x,y,z).
 * @param resolution         NumPy array float[3] voxel sizes (dx,dy,dz).
 * @param d_geo_grid         NumPy 3D array (float32) containing precomputed geometric distances.
 * @param source_position    NumPy array float[3] source position (world coords).
 * @param source_v_x         NumPy array float[3] source basis vector X.
 * @param source_v_y         NumPy array float[3] source basis vector Y (plane normal).
 * @param source_v_z         NumPy array float[3] source basis vector Z.
 * @param source_sad         Source-to-axis distance used in rescaling.
 * @param pri_s, pri_x,y,z   Primary-map scale/offset parameters (pri_z used in inverse-square).
 * @param sec_s, sec_x,y,z   Secondary-map scale/offset parameters.
 * @param samples            Supersampling factor per axis (1 = no supersampling).
 *
 * @note The wrapper creates temporary device-side textures and buffers and
 *       destroys them before returning. For repeated calls with the same
 *       fluence maps it is more efficient to create persistent texture
 *       objects and call a variant that accepts texture handles directly.
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
    float *d_terma_grid, *d_energies, *d_energy_weights, *d_mu_w, *d_off_axis_softening_fs_interp;
    int* d_num_voxels;
    cudaMalloc(&d_terma_grid, terma_grid_info.size * sizeof(float));
    cudaMalloc(&d_num_voxels, num_voxels_info.size * sizeof(int));
    cudaMalloc(&d_energies, energies_info.size * sizeof(float));
    cudaMalloc(&d_energy_weights, energy_weights_info.size * sizeof(float));
    cudaMalloc(&d_mu_w, mu_w_info.size * sizeof(float));
    cudaMalloc(&d_off_axis_softening_fs_interp, off_axis_softening_fs_interp_info.size * sizeof(float));
    cudaMemcpy(d_terma_grid, terma_grid_ptr, terma_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_voxels, num_voxels_ptr, num_voxels_info.size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energies, energies_ptr, energies_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energy_weights, energy_weights_ptr, energy_weights_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu_w, mu_w_ptr, mu_w_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_off_axis_softening_fs_interp, off_axis_softening_fs_interp_ptr, off_axis_softening_fs_interp_info.size * sizeof(float), cudaMemcpyHostToDevice);

    // Create texture object from host density array (helper in texture_utils.cuh)
    Texture3DHandle fluence_tex_handle = create_texture3d_from_ptr(fluence_grid_ptr, num_voxels_ptr[0], num_voxels_ptr[1], num_voxels_ptr[2], cudaFilterModeLinear);
    cudaTextureObject_t fluence_tex = fluence_tex_handle.tex;
    Texture3DHandle d_geo_tex_handle = create_texture3d_from_ptr(d_geo_grid_ptr, num_voxels_ptr[0], num_voxels_ptr[1], num_voxels_ptr[2], cudaFilterModeLinear);
    cudaTextureObject_t d_geo_tex = d_geo_tex_handle.tex;
    Texture3DHandle d_eff_tex_handle = create_texture3d_from_ptr(d_eff_grid_ptr, num_voxels_ptr[0], num_voxels_ptr[1], num_voxels_ptr[2], cudaFilterModeLinear);
    cudaTextureObject_t d_eff_tex = d_eff_tex_handle.tex;
    Texture3DHandle oad_tex_handle = create_texture3d_from_ptr(oad_grid_ptr, num_voxels_ptr[0], num_voxels_ptr[1], num_voxels_ptr[2], cudaFilterModeLinear);
    cudaTextureObject_t oad_tex = oad_tex_handle.tex;

    // Launch kernel
    dim3 dimBlock(16, 4, 4);
    dim3 dimGrid((num_voxels_ptr[0] + dimBlock.x - 1) / dimBlock.x,
        (num_voxels_ptr[1] + dimBlock.y - 1) / dimBlock.y,
        (num_voxels_ptr[2] + dimBlock.z - 1) / dimBlock.z);

    // Pass texture objects for the fluence maps and d_geo
    terma<<<dimGrid, dimBlock>>>(
        d_terma_grid,
        fluence_tex,
        d_geo_tex,
        d_eff_tex,
        d_num_voxels,
        num_energies,
        d_energies,
        d_energy_weights,
        d_mu_w,
        source_sad,
        oad_tex,
        d_off_axis_softening_fs_interp,
        off_axis_softening_dx);

    // Copy result back to host
    cudaMemcpy(terma_grid_ptr, d_terma_grid,
        terma_grid_info.size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_terma_grid);
    cudaFree(d_num_voxels);
    cudaFree(d_energies);
    cudaFree(d_energy_weights);
    cudaFree(d_mu_w);
    cudaFree(d_off_axis_softening_fs_interp);

    // Destroy textures and free CUDA arrays (helpers)
    destroy_texture3d(fluence_tex_handle);
    destroy_texture3d(d_geo_tex_handle);
    destroy_texture3d(d_eff_tex_handle);
    destroy_texture3d(oad_tex_handle);
}
