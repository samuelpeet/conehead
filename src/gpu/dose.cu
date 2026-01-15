#include "texture_utils.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

/**
 * @brief CUDA kernel that computes dose by collapsed-cone gather convolution.
 *
 * This __global__ kernel performs a gather-style collapsed-cone convolution of
 * TERMA: for each target voxel it marches along a fixed set of cone directions
 * and accumulates contributions sampled from TERMA using a precomputed
 * angular-depth kernel. The final voxel value is rescaled using the
 * second step of the "no-tilt" approximation to reintroduce the inverse-square
 * falloff removed earlier in `terma()`.
 *
 * Behaviour summary:
 *  - One thread computes one target voxel (flattened index idx = x + y*nx + z*nx*ny).
 *  - If mask_grid[idx] == 0.0f the kernel skips convolution and sets dose = terma at that voxel.
 *  - Otherwise the kernel loops over theta/phi sample directions, marches along each ray
 *    in steps of `ds_cm`, accumulates radiological depth from `density_grid`, looks up the
 *    kernel value for the corresponding radiological depth and accumulates terma*kernel*vol.
 *  - Ray tracing stops when the ray exits the grid, the radiological depth exceeds kernel support,
 *    or the travelled distance exceeds `max_kernel_depth_cm`.
 *
 * Important implementation notes:
 *  - Kernel values are cumulative: kernel[phi, depth_i] = integral from depth 0 to depth_i.
 *  - Dose contribution in each shell is the difference: delta = kernel[depth_i] - kernel[depth_i-1].
 *  - kernel_omegas contains per-phi solid-angle weights; sample contribution volume is
 *    kernel_omegas[ip] * ds_cm * s * s, where s is the distance travelled along the ray.
 *  - All 3D grids use flattened row-major indexing.
 *
 * @param[in,out] dose_grid         Device pointer to flattened float array (nx*ny*nz) where dose is written.
 * @param[in]  resolution        Device pointer to float[3] voxel sizes (dx,dy,dz).
 * @param[in]  num_voxels        Device pointer to int[3] = {nx,ny,nz}.
 * @param[in]  corner            Device pointer to float[3] world-space grid corner (min x,y,z).
 * @param[in]  density_grid      Device pointer to flattened density grid (nx*ny*nz).
 * @param[in]  d_eff_grid        Device pointer to flattened radiological depth grid (nx*ny*nz).
 * @param[in]  d_geo_grid        Device pointer to geometric distances (nx*ny*nz) used for final rescaling.
 * @param[in]  terma_grid       Device pointer to flattened TERMA grid (nx*ny*nz) (output of terma()).
 * @param[in]  mask_grid         Device pointer to flattened mask (0.0/1.0) used to skip work when zero.
 * @param[in]  kernel_thetas     Device pointer to array of theta angles (degrees), length n_thetas (16).
 * @param[in]  kernel_phis       Device pointer to array of phi angles (degrees), length n_phis (12).
 * @param[in]  kernel_omegas     Device pointer to per-phi solid-angle weights, length n_phis.
 * @param[in]  kernel            Device pointer to flattened cumulative kernel values sized n_spectrum_depth_bins * n_phis * n_depth_bins
 *                               (layout [spectrum_depth][phi][depth]).
 * @param[in]  source_sad        Source-to-axis distance used for no-tilt rescaling.
 * @param[in]  source_position   Device pointer to float[3] (not used for alignment in current impl.).
 * @param[in]  source_v_x/y/z    Device pointers to float[3] source basis vectors (present for context).
 * @param[in]  n_depth_bins      Number of radial depth samples per kernel (kernel depth axis length).
 * @param[in]  n_spectrum_depth_bins Number of spectrum-hardening depth bins.
 * @param[in]  kernel_depth_res_cm Depth resolution of kernel radial bins (cm).
 * @param[in]  max_kernel_depth_cm Maximum kernel radial depth to march (cm).
 * @param[in]  spectrum_depth_res_cm Depth resolution for spectrum-hardening bins (cm).
 * @param[in]  max_spectrum_depth_cm Maximum spectrum-hardening depth captured (cm).
 * @param[in]  spectrum_hardening_enable Flag to enable/disable spectrum hardening (true=enabled, false=use surface spectrum).
 * @param[in]  ds_cm             Ray-marching step size (cm).
 */
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
    int spectrum_hardening_enable,
    float ds_cm)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = num_voxels[0];
    int ny = num_voxels[1];
    int nz = num_voxels[2];

    int idx = x + y * nx + z * nx * ny;

    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    // TERMA mask
    if (mask_grid[idx] == 0.0f) {
        // Skip convolution for this voxel
        dose_grid[idx] = 0.0f;
        return;
    }

    // Voxel geometry
    float3 resolution_f3 = make_float3(resolution[0], resolution[1], resolution[2]);
    float3 corner_f3 = make_float3(corner[0], corner[1], corner[2]);
    float3 centre_f3 = make_float3(
        corner_f3.x + resolution_f3.x * (x + 0.5),
        corner_f3.y + resolution_f3.y * (y + 0.5),
        corner_f3.z + resolution_f3.z * (z + 0.5));

    // Outside external/support structures
    float density = tex3D<float>(density_tex, x + 0.5f, y + 0.5f, z + 0.5f);
    if (density == 0.0f) {
        // Skip convolution for this voxel
        dose_grid[idx] = 0.0f;
        return;
    }

    // Read source basis vectors (assumed to be unit / orthonormal)
    float3 svx = make_float3(source_v_x[0], source_v_x[1], source_v_x[2]);
    float3 svy = make_float3(source_v_y[0], source_v_y[1], source_v_y[2]);
    float3 svz = make_float3(source_v_z[0], source_v_z[1], source_v_z[2]);

    const float kernel_depth_res_cm_inv = 1.0f / kernel_depth_res_cm;
    const float spectrum_depth_res_cm_inv = 1.0f / spectrum_depth_res_cm;

    // Precompute trigonometric values for all thetas and phis
    const float deg_to_rad = 3.141592653589793 / 180.0;
    const int n_thetas = 16;
    const int n_phis = 12;
    float theta_rad_arr[n_thetas];
    float phi_rad_arr[n_phis];
    // float c_t_arr[n_thetas];
    // float s_t_arr[n_thetas];
    // float c_p_arr[n_phis];
    // float s_p_arr[n_phis];
    for (int it = 0; it < n_thetas; it++) {
        theta_rad_arr[it] = kernel_thetas[it] * deg_to_rad;
        // c_t_arr[it] = cos(theta_rad_arr[it]);
        // s_t_arr[it] = sin(theta_rad_arr[it]);
    }
    for (int ip = 0; ip < n_phis; ip++) {
        phi_rad_arr[ip] = (kernel_phis[ip] - 180.0) * deg_to_rad;
        // c_p_arr[ip] = cos(phi_rad_arr[ip]);
        // s_p_arr[ip] = sin(phi_rad_arr[ip]);
    }

    const int kernel_phi_stride = n_depth_bins;
    const int kernel_depth_stride = n_phis * kernel_phi_stride;
    const int max_spectrum_bin = n_spectrum_depth_bins - 1;

    // For use in the ray marching loop
    float3 direction_f3 = make_float3(0.0f, 0.0f, 0.0f);
    float3 position_f3 = make_float3(0.0f, 0.0f, 0.0f);
    float dose_acc = 0.0f;
    float res_x_inv = __frcp_rn(resolution_f3.x);
    float res_y_inv = __frcp_rn(resolution_f3.y);
    float res_z_inv = __frcp_rn(resolution_f3.z);
    int max_steps = __float2int_rd(max_kernel_depth_cm / ds_cm);
    int nx_ny = nx * ny;

    int spectrum_idx = 0;
    if (spectrum_hardening_enable) {
        // Precompute spectrum-depth index once per target voxel (branchless clamp, floor)
        float depth_wet_centre = __ldg(&d_eff_grid[idx]);
        float depth_bin_f_c = fminf(depth_wet_centre, max_spectrum_depth_cm) * spectrum_depth_res_cm_inv;
        int spectrum_idx_c = __float2int_rn(depth_bin_f_c);
        float spectrum_idx_c_f = fminf(fmaxf((float)spectrum_idx_c, 0.0f), (float)max_spectrum_bin);
        spectrum_idx = __float2int_rn(spectrum_idx_c_f);
    }
    int kernel_s_base = spectrum_idx * kernel_depth_stride;

#pragma unroll(1)
    for (int it = 0; it < n_thetas; it++) {
#pragma unroll(1)
        for (int ip = 0; ip < n_phis; ip++) {

            // Initialize ray position at voxel centre
            position_f3.x = centre_f3.x;
            position_f3.y = centre_f3.y;
            position_f3.z = centre_f3.z;

            float local_x = cos(theta_rad_arr[it]) * sin(phi_rad_arr[ip]); // local frame x
            float local_y = cos(phi_rad_arr[ip]); // local frame y
            float local_z = sin(theta_rad_arr[it]) * sin(phi_rad_arr[ip]); // local frame z

            // Transform local/source-frame direction into world coordinates using basis vectors
            direction_f3.x = local_x * svx.x + local_y * svy.x + local_z * svz.x;
            direction_f3.y = local_x * svx.y + local_y * svy.y + local_z * svz.y;
            direction_f3.z = local_x * svx.z + local_y * svy.z + local_z * svz.z;

            float s = 0.0f; // Geometrtic distance travelled along ray (cm)
            float rad_depth = 0.0f; // Effective water equivalent distance travelled along ray (cm)
            float rad_depth_prev = 0.0f;
            float omega = kernel_omegas[ip];
            float kernel_value_prev = 0.0f; // Previous cumulative kernel value (initialize to 0 at ray start)

            // We have direction and starting position; time to march along ray
            for (int step = 0; step < max_steps; step++) {

                // Advance position FIRST (we're computing dose at the target, 
                // accumulated from sources along the ray)
                position_f3.x += direction_f3.x * ds_cm;
                position_f3.y += direction_f3.y * ds_cm;
                position_f3.z += direction_f3.z * ds_cm;
                s += ds_cm;

                // Map current step position → voxel indices
                float fx = (position_f3.x - corner_f3.x) * res_x_inv;
                float fy = (position_f3.y - corner_f3.y) * res_y_inv;
                float fz = (position_f3.z - corner_f3.z) * res_z_inv;
                int ix = __float2int_rd(fx);
                int iy = __float2int_rd(fy);
                int iz = __float2int_rd(fz);
                if ((unsigned)ix >= nx || (unsigned)iy >= ny || (unsigned)iz >= nz) {
                    break; // Ray left grid
                }
                int idx2 = ix + iy * nx + iz * nx_ny;

                // Sample density and TERMA at the NEW position (after stepping)
                float rho = tex3D<float>(density_tex, fx + 0.5f, fy + 0.5f, fz + 0.5f);
                if(rho == 0.0f) {
                    continue;
                }
                float terma = __ldg(&terma_grid[idx2]);

                // Compute no-tilt approximation inverse-square law terma rescaling at the new position
                float d_geo = __ldg(&d_geo_grid[idx2]);
                terma *= (source_sad / d_geo) * (source_sad / d_geo);

                // Accumulate radiological depth    
                rad_depth += rho * ds_cm;                
                if (rad_depth >= max_kernel_depth_cm) {
                    break; // Beyond kernel support
                }

                // Lookup kernel index for the interval we just traversed
                int depth_idx = __float2int_rn(rad_depth * kernel_depth_res_cm_inv) - 1;
                if (depth_idx < 0) {
                    continue;  // Haven't reached first kernel bin yet
                }
                int kernel_base = kernel_s_base + ip * kernel_phi_stride;
                float kernel_value_curr = kernel[kernel_base + depth_idx];
                float kernel_value_diff = kernel_value_curr - kernel_value_prev;
                kernel_value_prev = kernel_value_curr;
                // float kernel_value_diff = kernel[kernel_base + depth_idx];

                // Volume element for this step (using GEOMETRIC distance)
                float d_r = (s * s * s) - ((s - ds_cm) * (s - ds_cm) * (s - ds_cm));
                float vol = omega * d_r / 3.0f;
                rad_depth_prev = rad_depth;

                dose_acc += terma * kernel_value_diff * vol;
            }
        }
    }

    // Store final dose
    dose_grid[idx] = dose_acc;
}

/**
 * @brief Python-visible host wrapper that prepares inputs and launches the `dose` kernel.
 *
 * This function is bound via pybind11 and accepts NumPy arrays for the output
 * dose grid, voxel geometry, density/terma fields, mask and the angular kernel
 * data. It copies inputs to device buffers, launches the `dose` kernel and
 * copies the computed dose back into the provided `dose_grid` array.
 *
 * Requirements and notes:
 *  - All NumPy arrays must be C-contiguous. Most arrays must be float32; `num_voxels`
 *    must be int32. `dose_grid` is overwritten in-place.
 *  - The wrapper performs synchronous host↔device copies on the default stream.
 *  - For repeated calls consider reusing device allocations or exposing a
 *    persistent texture/kernel upload API to avoid allocation overhead.
 *
 * @param[in,out] dose_grid      NumPy array (float32, flattened) preallocated for nx*ny*nz elements.
 * @param[in]     resolution     NumPy array float[3] voxel sizes (dx,dy,dz).
 * @param[in]     num_voxels     NumPy array int[3] = {nx,ny,nz}.
 * @param[in]     corner         NumPy array float[3] world-space grid corner (min x,y,z).
 * @param[in]     density_grid   NumPy array (float32) flattened density grid.
 * @param[in]     d_geo_grid     NumPy array (float32) flattened geometric distances.
 * @param[in]     terma_grid     NumPy array (float32) flattened TERMA grid.
 * @param[in]     mask_grid      NumPy array (float32) flattened mask (0.0/1.0) used to skip work.
 * @param[in]     kernel_thetas  NumPy array (float32) theta angles (degrees), length n_thetas (16).
 * @param[in]     kernel_phis    NumPy array (float32) phi angles (degrees), length n_phis (12).
 * @param[in]     kernel_omegas  NumPy array (float32) per-phi solid-angle weights.
 * @param[in]     kernel         NumPy array (float32) flattened kernel values (n_phis * n_depth_bins).
 * @param[in]     source_sad     Float: source-to-axis distance used for no-tilt rescaling.
 * @param[in]     source_position NumPy array float[3] source position (world coords).
 * @param[in]     source_v_x/v_y/v_z NumPy arrays float[3] source basis vectors.
 * @param[in]     n_depth_bins   Int: number of depth bins in the kernel.
 * @param[in]     kernel_depth_res_cm Float: depth resolution of kernel bins (cm).
 * @param[in]     max_kernel_depth_cm  Float: maximum kernel depth to march (cm).
 * @param[in]     ds_cm          Float: ray-marching step size (cm).
 */
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
    float ds_cm)
{
    pybind11::buffer_info dose_grid_info = dose_grid.request();
    pybind11::buffer_info resolution_info = resolution.request();
    pybind11::buffer_info num_voxels_info = num_voxels.request();
    pybind11::buffer_info corner_info = corner.request();
    pybind11::buffer_info density_grid_info = density_grid.request();
    pybind11::buffer_info d_eff_grid_info = d_eff_grid.request();
    pybind11::buffer_info d_geo_grid_info = d_geo_grid.request();
    pybind11::buffer_info terma_grid_info = terma_grid.request();
    pybind11::buffer_info mask_grid_info = mask_grid.request();
    pybind11::buffer_info kernel_thetas_info = kernel_thetas.request();
    pybind11::buffer_info kernel_phis_info = kernel_phis.request();
    pybind11::buffer_info kernel_omegas_info = kernel_omegas.request();
    pybind11::buffer_info kernel_info = kernel.request();
    pybind11::buffer_info source_position_info = source_position.request();
    pybind11::buffer_info source_v_x_info = source_v_x.request();
    pybind11::buffer_info source_v_y_info = source_v_y.request();
    pybind11::buffer_info source_v_z_info = source_v_z.request();

    float* dose_grid_ptr = reinterpret_cast<float*>(dose_grid_info.ptr);
    float* resolution_ptr = reinterpret_cast<float*>(resolution_info.ptr);
    int* num_voxels_ptr = reinterpret_cast<int*>(num_voxels_info.ptr);
    float* corner_ptr = reinterpret_cast<float*>(corner_info.ptr);
    float* density_grid_ptr = reinterpret_cast<float*>(density_grid_info.ptr);
    float* d_eff_grid_ptr = reinterpret_cast<float*>(d_eff_grid_info.ptr);
    float* d_geo_grid_ptr = reinterpret_cast<float*>(d_geo_grid_info.ptr);
    float* terma_grid_ptr = reinterpret_cast<float*>(terma_grid_info.ptr);
    float* mask_grid_ptr = reinterpret_cast<float*>(mask_grid_info.ptr);
    float* kernel_thetas_ptr = reinterpret_cast<float*>(kernel_thetas_info.ptr);
    float* kernel_phis_ptr = reinterpret_cast<float*>(kernel_phis_info.ptr);
    float* kernel_omegas_ptr = reinterpret_cast<float*>(kernel_omegas_info.ptr);
    float* kernel_ptr = reinterpret_cast<float*>(kernel_info.ptr);
    float* source_position_ptr = reinterpret_cast<float*>(source_position_info.ptr);
    float* source_v_x_ptr = reinterpret_cast<float*>(source_v_x_info.ptr);
    float* source_v_y_ptr = reinterpret_cast<float*>(source_v_y_info.ptr);
    float* source_v_z_ptr = reinterpret_cast<float*>(source_v_z_info.ptr);

    // Allocate device memory and copy inputs
    float *d_dose_grid, *d_resolution, *d_corner, *d_d_eff_grid, *d_d_geo_grid; //, *d_density_grid;
    float *d_terma_grid, *d_mask_grid;
    float *d_kernel_thetas, *d_kernel_phis, *d_kernel_omegas, *d_kernel;
    float *d_source_position, *d_source_v_x, *d_source_v_y, *d_source_v_z;
    int* d_num_voxels;
    cudaMalloc(&d_dose_grid, dose_grid_info.size * sizeof(float));
    cudaMalloc(&d_resolution, resolution_info.size * sizeof(float));
    cudaMalloc(&d_num_voxels, num_voxels_info.size * sizeof(int));
    cudaMalloc(&d_corner, corner_info.size * sizeof(float));
    // cudaMalloc(&d_density_grid, density_grid_info.size * sizeof(float));
    cudaMalloc(&d_d_eff_grid, d_eff_grid_info.size * sizeof(float));
    cudaMalloc(&d_d_geo_grid, d_geo_grid_info.size * sizeof(float));
    cudaMalloc(&d_terma_grid, terma_grid_info.size * sizeof(float));
    cudaMalloc(&d_mask_grid, mask_grid_info.size * sizeof(float));
    cudaMalloc(&d_kernel_thetas, kernel_thetas_info.size * sizeof(float));
    cudaMalloc(&d_kernel_phis, kernel_phis_info.size * sizeof(float));
    cudaMalloc(&d_kernel_omegas, kernel_omegas_info.size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_info.size * sizeof(float));
    cudaMalloc(&d_source_position, source_position_info.size * sizeof(float));
    cudaMalloc(&d_source_v_x, source_v_x_info.size * sizeof(float));
    cudaMalloc(&d_source_v_y, source_v_y_info.size * sizeof(float));
    cudaMalloc(&d_source_v_z, source_v_z_info.size * sizeof(float));
    cudaMemcpy(d_dose_grid, dose_grid_ptr, dose_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_resolution, resolution_ptr, resolution_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_voxels, num_voxels_ptr, num_voxels_info.size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_corner, corner_ptr, corner_info.size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_density_grid, density_grid_ptr, density_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_eff_grid, d_eff_grid_ptr, d_eff_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_geo_grid, d_geo_grid_ptr, d_geo_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_terma_grid, terma_grid_ptr, terma_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask_grid, mask_grid_ptr, mask_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_thetas, kernel_thetas_ptr, kernel_thetas_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_phis, kernel_phis_ptr, kernel_phis_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_omegas, kernel_omegas_ptr, kernel_omegas_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel_ptr, kernel_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_position, source_position_ptr, source_position_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_v_x, source_v_x_ptr, source_v_x_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_v_y, source_v_y_ptr, source_v_y_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_v_z, source_v_z_ptr, source_v_z_info.size * sizeof(float), cudaMemcpyHostToDevice);

    // Create texture object from host density array (helper in texture_utils.cuh)
    Texture3DHandle texh = create_texture3d_from_ptr(density_grid_ptr, num_voxels_ptr[0], num_voxels_ptr[1], num_voxels_ptr[2], cudaFilterModeLinear);
    cudaTextureObject_t density_tex = texh.tex;

    // Launch kernel
    dim3 dimBlock(16, 4, 4);
    dim3 dimGrid((num_voxels_ptr[0] + dimBlock.x - 1) / dimBlock.x,
        (num_voxels_ptr[1] + dimBlock.y - 1) / dimBlock.y,
        (num_voxels_ptr[2] + dimBlock.z - 1) / dimBlock.z);

    dose<<<dimGrid, dimBlock>>>(d_dose_grid, d_resolution, d_num_voxels, d_corner,
        density_tex, d_d_eff_grid, d_d_geo_grid, d_terma_grid, d_mask_grid,
        d_kernel_thetas, d_kernel_phis, d_kernel_omegas, d_kernel,
        source_sad, d_source_position, d_source_v_x, d_source_v_y, d_source_v_z,
        n_depth_bins, n_spectrum_depth_bins, kernel_depth_res_cm, max_kernel_depth_cm,
        spectrum_depth_res_cm, max_spectrum_depth_cm, spectrum_hardening_enable, ds_cm);

    // cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(dose_grid_ptr, d_dose_grid,
        dose_grid_info.size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_dose_grid);
    cudaFree(d_resolution);
    cudaFree(d_num_voxels);
    cudaFree(d_corner);
    cudaFree(d_d_eff_grid);
    cudaFree(d_d_geo_grid);
    cudaFree(d_terma_grid);
    cudaFree(d_mask_grid);
    cudaFree(d_kernel_thetas);
    cudaFree(d_kernel_phis);
    cudaFree(d_kernel_omegas);
    cudaFree(d_kernel);
    cudaFree(d_source_position);
    cudaFree(d_source_v_x);
    cudaFree(d_source_v_y);
    cudaFree(d_source_v_z);

    // Destroy texture and free CUDA array (helper)
    destroy_texture3d(texh);
}
