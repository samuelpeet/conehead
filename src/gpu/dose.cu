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
 *  - Angular sampling uses hard-coded counts (n_thetas=16, n_phis=12); kernel layout is [phi][depth].
 *  - kernel_omegas contains per-phi solid-angle weights; sample contribution volume is
 *    kernel_omegas[ip] * ds_cm * s * s, where s is the distance travelled along the ray.
 *  - All 3D grids use flattened row-major indexing.
 *
 * @param[in,out] dose_grid         Device pointer to flattened float array (nx*ny*nz) where dose is written.
 * @param[in]  resolution        Device pointer to float[3] voxel sizes (dx,dy,dz).
 * @param[in]  num_voxels        Device pointer to int[3] = {nx,ny,nz}.
 * @param[in]  corner            Device pointer to float[3] world-space grid corner (min x,y,z).
 * @param[in]  density_grid      Device pointer to flattened density grid (nx*ny*nz).
 * @param[in]  d_geo_grid        Device pointer to geometric distances (nx*ny*nz) used for final rescaling.
 * @param[in]  terma_grid       Device pointer to flattened TERMA grid (nx*ny*nz) (output of terma()).
 * @param[in]  mask_grid         Device pointer to flattened mask (0.0/1.0) used to skip work when zero.
 * @param[in]  kernel_thetas     Device pointer to array of theta angles (degrees), length n_thetas (16).
 * @param[in]  kernel_phis       Device pointer to array of phi angles (degrees), length n_phis (12).
 * @param[in]  kernel_omegas     Device pointer to per-phi solid-angle weights, length n_phis.
 * @param[in]  kernel            Device pointer to flattened kernel values sized n_phis * n_depth_bins (layout [phi][depth]).
 * @param[in]  source_sad        Source-to-axis distance used for no-tilt rescaling.
 * @param[in]  source_position   Device pointer to float[3] (not used for alignment in current impl.).
 * @param[in]  source_v_x/y/z    Device pointers to float[3] source basis vectors (present for context).
 * @param[in]  n_depth_bins      Number of depth samples per kernel (kernel depth axis length).
 * @param[in]  kernel_depth_res_cm Depth resolution of kernel bins (cm).
 * @param[in]  max_kernel_depth_cm Maximum kernel depth to march (cm).
 * @param[in]  ds_cm             Ray-marching step size (cm).
 */
__global__ void dose(float* dose_grid,
    float* resolution,
    int* num_voxels,
    float* corner,
    float* density_grid,
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
    float kernel_depth_res_cm,
    float max_kernel_depth_cm,
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

    if (mask_grid[idx] == 0.0f) {
        // Skip convolution for this voxel
        dose_grid[idx] = terma_grid[idx];
        return;
    }

    // Voxel geometry
    float3 resolution_f3 = make_float3(resolution[0], resolution[1], resolution[2]);
    float3 corner_f3 = make_float3(corner[0], corner[1], corner[2]);
    float3 centre_f3 = make_float3(
        corner_f3.x + resolution_f3.x * (x + 0.5),
        corner_f3.y + resolution_f3.y * (y + 0.5),
        corner_f3.z + resolution_f3.z * (z + 0.5));

    // Read source basis vectors (assumed to be unit / orthonormal)
    float3 svx = make_float3(source_v_x[0], source_v_x[1], source_v_x[2]);
    float3 svy = make_float3(source_v_y[0], source_v_y[1], source_v_y[2]);
    float3 svz = make_float3(source_v_z[0], source_v_z[1], source_v_z[2]);

    // Precompute trigonometric values for all thetas and phis
    const int n_thetas = 16;
    const int n_phis = 12;
    float theta_rad_arr[n_thetas];
    float phi_rad_arr[n_phis];
    float c_t_arr[n_thetas];
    float s_t_arr[n_thetas];
    float c_p_arr[n_phis];
    float s_p_arr[n_phis];
    for (int it = 0; it < n_thetas; it++) {
        theta_rad_arr[it] = kernel_thetas[it] * 3.141592653589793 / 180.0;
        c_t_arr[it] = cos(theta_rad_arr[it]);
        s_t_arr[it] = sin(theta_rad_arr[it]);
    }
    for (int ip = 0; ip < n_phis; ip++) {
        phi_rad_arr[ip] = (kernel_phis[ip] - 180.0) * 3.141592653589793 / 180.0;
        c_p_arr[ip] = cos(phi_rad_arr[ip]);
        s_p_arr[ip] = sin(phi_rad_arr[ip]);
    }

    // For use in the ray marching loop
    float3 direction_f3 = make_float3(0.0f, 0.0f, 0.0f);
    float3 position_f3 = make_float3(0.0f, 0.0f, 0.0f);
    float dose_acc = 0.0f;

    for (int it = 0; it < n_thetas; it++) {
        for (int ip = 0; ip < n_phis; ip++) {
            float s = 0.0f;
            float rad_depth = 0.0f;
            int max_steps = (int)(max_kernel_depth_cm / ds_cm);

            // Initialize ray position at voxel centre
            position_f3.x = centre_f3.x;
            position_f3.y = centre_f3.y;
            position_f3.z = centre_f3.z;

            float local_x = c_t_arr[it] * s_p_arr[ip]; // local frame x
            float local_y = c_p_arr[ip]; // local frame y
            float local_z = s_t_arr[it] * s_p_arr[ip]; // local frame z

            // Transform local/source-frame direction into world coordinates using basis vectors
            direction_f3.x = local_x * svx.x + local_y * svy.x + local_z * svz.x;
            direction_f3.y = local_x * svx.y + local_y * svy.y + local_z * svz.y;
            direction_f3.z = local_x * svx.z + local_y * svy.z + local_z * svz.z;

            // Normalize
            float mag = sqrt(direction_f3.x * direction_f3.x + direction_f3.y * direction_f3.y + direction_f3.z * direction_f3.z);
            direction_f3.x /= mag;
            direction_f3.y /= mag;
            direction_f3.z /= mag;

            // We have direction and starting position; time to march along ray
            for (int step = 0; step < max_steps; step++) {
                // Advance a step
                position_f3.x += direction_f3.x * ds_cm;
                position_f3.y += direction_f3.y * ds_cm;
                position_f3.z += direction_f3.z * ds_cm;
                s += ds_cm;

                // Map position → voxel indices
                int ix = (int)((position_f3.x - corner_f3.x) / resolution_f3.x);
                int iy = (int)((position_f3.y - corner_f3.y) / resolution_f3.y);
                int iz = (int)((position_f3.z - corner_f3.z) / resolution_f3.z);
                int idx2 = ix + iy * nx + iz * nx * ny;
                if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
                    break; // Ray left grid
                }
                // Accumulate radiological depth
                rad_depth += density_grid[idx2] * ds_cm;

                // Lookup kernel value corresponding to this radiological depth
                int depth_idx = (int)(rad_depth / kernel_depth_res_cm);
                if (depth_idx >= n_depth_bins) {
                    break; // Beyond end of kernel
                }
                float kernel_value = kernel[ip * n_depth_bins + depth_idx];
                float vol = kernel_omegas[ip] * ds_cm * s * s; // Volume of sample sector
                dose_acc += terma_grid[idx2] * kernel_value * vol;
                if (s >= max_kernel_depth_cm) {
                    break;
                }
            }
        }
    }
    // Compute no-tilt approximation inverse-square law rescaling
    float no_tilt_rescaling = (source_sad / d_geo_grid[idx]) * (source_sad / d_geo_grid[idx]);

    // Store final dose
    dose_grid[idx] = dose_acc * no_tilt_rescaling;
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
    float kernel_depth_res_cm,
    float max_kernel_depth_cm,
    float ds_cm)
{
    pybind11::buffer_info dose_grid_info = dose_grid.request();
    pybind11::buffer_info resolution_info = resolution.request();
    pybind11::buffer_info num_voxels_info = num_voxels.request();
    pybind11::buffer_info corner_info = corner.request();
    pybind11::buffer_info density_grid_info = density_grid.request();
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
    float *d_dose_grid, *d_resolution, *d_corner, *d_density_grid, *d_d_geo_grid;
    float *d_terma_grid, *d_mask_grid;
    float *d_kernel_thetas, *d_kernel_phis, *d_kernel_omegas, *d_kernel;
    float *d_source_position, *d_source_v_x, *d_source_v_y, *d_source_v_z;
    int* d_num_voxels;
    cudaMalloc(&d_dose_grid, dose_grid_info.size * sizeof(float));
    cudaMalloc(&d_resolution, resolution_info.size * sizeof(float));
    cudaMalloc(&d_num_voxels, num_voxels_info.size * sizeof(int));
    cudaMalloc(&d_corner, corner_info.size * sizeof(float));
    cudaMalloc(&d_density_grid, density_grid_info.size * sizeof(float));
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
    cudaMemcpy(d_density_grid, density_grid_ptr, density_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
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

    // Launch kernel
    dim3 dimBlock(16, 4, 4);
    dim3 dimGrid((num_voxels_ptr[0] + dimBlock.x - 1) / dimBlock.x,
        (num_voxels_ptr[1] + dimBlock.y - 1) / dimBlock.y,
        (num_voxels_ptr[2] + dimBlock.z - 1) / dimBlock.z);

    dose<<<dimGrid, dimBlock>>>(d_dose_grid, d_resolution, d_num_voxels, d_corner,
        d_density_grid, d_d_geo_grid, d_terma_grid, d_mask_grid,
        d_kernel_thetas, d_kernel_phis, d_kernel_omegas, d_kernel,
        source_sad, d_source_position, d_source_v_x, d_source_v_y, d_source_v_z,
        n_depth_bins, kernel_depth_res_cm, max_kernel_depth_cm, ds_cm);

    // Copy result back to host
    cudaMemcpy(dose_grid_ptr, d_dose_grid,
        dose_grid_info.size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_dose_grid);
    cudaFree(d_resolution);
    cudaFree(d_num_voxels);
    cudaFree(d_corner);
    cudaFree(d_density_grid);
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
}
