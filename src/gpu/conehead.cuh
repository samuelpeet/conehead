#ifndef CONEHEAD_GPU_CONEHEAD_CUH
#define CONEHEAD_GPU_CONEHEAD_CUH

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// CUDA kernels (device entry points)
__global__ void d_geo(float* d_geo_grid,
    int* num_voxels,
    float* corner,
    float* resolution,
    float* source_position);

__global__ void oad(float* oad_grid,
    int* num_voxels,
    float* corner,
    float* resolution,
    float* source_position,
    float* source_v_x,
    float* source_v_y,
    float* source_v_z,
    float* source_isocenter);

__global__ void d_eff(float* d_eff_grid,
    int* num_voxels,
    float* corner,
    float* resolution,
    cudaTextureObject_t density_tex,
    float* source_position);

__global__ void fluence(float* fluence_grid,
    cudaTextureObject_t fluence_map_pri,
    cudaTextureObject_t fluence_map_sec,
    int* num_voxels,
    float* corner,
    float* resolution,
    cudaTextureObject_t d_geo_tex,
    float* source_position,
    float* source_v_x,
    float* source_v_y,
    float* source_v_z,
    float* source_isocenter,
    float source_sad,
    float pri_z,
    float sec_z,
    int samples);

__global__ void terma(float* terma_grid,
    float* fluence_grid,
    float* d_geo_grid,
    float* d_eff_grid,
    int* num_voxels,
    int num_energies,
    float* energies,
    float* energy_weights,
    float* mu_tot,
    float* mu_en,
    float source_sad,
    float* oad_grid,
    float* off_axis_softening_fs_interp,
    float off_axis_softening_dx,
    float off_axis_softening_oad_max);

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
    float source_sad);

__global__ void mask(float* mask_grid,
    float* terma_grid,
    int* num_voxels,
    float* resolution,
    float max_distance_cm,
    float terma_threshold);

__global__ void dose(float* dose_grid,
    float* resolution,
    int* num_voxels,
    float* corner,
    cudaTextureObject_t density_tex,
    float* d_eff_grid,
    float* d_geo_grid,
    float* terma_grid,
    float* mask_grid,
    float* kernel_thetas,
    float* kernel_phis,
    float* kernel_omegas,
    float* kernel,
    float source_sad,
    float* source_position,
    float* source_v_x,
    float* source_v_y,
    float* source_v_z,
    int n_depth_bins,
    int n_spectrum_depth_bins,
    float kernel_depth_res_cm,
    float max_kernel_depth_cm,
    float spectrum_depth_res_cm,
    float max_spectrum_depth_cm,
    bool spectrum_hardening_enable,
    float ds_cm);

// Host-facing wrapper functions (implemented in the corresponding .cu files).
// These are the functions exposed to Python via pybind11 in bindings.cu.
void map_d_geo(pybind11::array_t<float> d_geo_grid, pybind11::array_t<int> num_voxels,
    pybind11::array_t<float> corner, pybind11::array_t<float> resolution,
    pybind11::array_t<float> source_position);

void map_oad(pybind11::array_t<float> oad_grid, pybind11::array_t<int> num_voxels,
    pybind11::array_t<float> corner, pybind11::array_t<float> resolution,
    pybind11::array_t<float> source_position, pybind11::array_t<float> source_v_x,
    pybind11::array_t<float> source_v_y, pybind11::array_t<float> source_v_z,
    pybind11::array_t<float> source_isocenter);

void map_d_eff(pybind11::array_t<float> d_eff_grid, pybind11::array_t<int> num_voxels,
    pybind11::array_t<float> corner, pybind11::array_t<float> resolution,
    pybind11::array_t<float> density_grid, pybind11::array_t<float> source_position);

void map_fluence(pybind11::array_t<float> fluence_grid,
    pybind11::array_t<float> fluence_map_pri, pybind11::array_t<float> fluence_map_sec,
    pybind11::array_t<int> num_voxels, pybind11::array_t<float> corner,
    pybind11::array_t<float> resolution, pybind11::array_t<float> d_geo_grid,
    pybind11::array_t<float> source_position, pybind11::array_t<float> source_v_x, pybind11::array_t<float> source_v_y,
    pybind11::array_t<float> source_v_z, pybind11::array_t<float> source_isocenter, float source_sad,
    float pri_z, float sec_z,
    int samples);

void map_terma(
    pybind11::array_t<float> terma_grid,
    pybind11::array_t<float> fluence_grid,
    pybind11::array_t<float> d_geo_grid,
    pybind11::array_t<float> d_eff_grid,
    pybind11::array_t<int> num_voxels,
    int num_energies,
    pybind11::array_t<float> energies,
    pybind11::array_t<float> energy_weights,
    pybind11::array_t<float> mu_tot,
    pybind11::array_t<float> mu_en,
    float source_sad,
    pybind11::array_t<float> oad_grid,
    pybind11::array_t<float> off_axis_softening_fs_interp,
    float off_axis_softening_dx,
    float off_axis_softening_oad_max);

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
    float source_sad);

void map_mask(pybind11::array_t<float> mask_grid,
    pybind11::array_t<float> terma_grid,
    pybind11::array_t<int> num_voxels,
    pybind11::array_t<float> resolution,
    float max_distance_cm,
    float terma_threshold);

void map_dose(pybind11::array_t<float> dose_grid,
    pybind11::array_t<float> resolution,
    pybind11::array_t<int> num_voxels,
    pybind11::array_t<float> corner,
    pybind11::array_t<float> density_grid,
    pybind11::array_t<float> d_eff_grid,
    pybind11::array_t<float> d_geo_grid,
    pybind11::array_t<float> terma_grid,
    pybind11::array_t<float> mask_grid,
    pybind11::array_t<float> kernel_thetas,
    pybind11::array_t<float> kernel_phis,
    pybind11::array_t<float> kernel_omegas,
    pybind11::array_t<float> kernel,
    float source_sad,
    pybind11::array_t<float> source_position,
    pybind11::array_t<float> source_v_x,
    pybind11::array_t<float> source_v_y,
    pybind11::array_t<float> source_v_z,
    int n_depth_bins,
    int n_spectrum_depth_bins,
    float kernel_depth_res_cm,
    float max_kernel_depth_cm,
    float spectrum_depth_res_cm,
    float max_spectrum_depth_cm,
    bool spectrum_hardening_enable,
    float ds_cm);

#endif // CONEHEAD_GPU_CONEHEAD_CUH
