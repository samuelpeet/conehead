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
 * @brief Compute per-voxel fluence by projecting voxel samples onto fluence planes.
 *
 * This GPU kernel projects supersampled voxel subpoints onto two fluence
 * planes (primary and secondary), looks up map values from 2D texture objects,
 * applies inverse-square scaling using a precomputed geometric-distance 3D
 * texture and writes the total fluence into the flattened output grid.
 *
 * Steps performed for each (x,y,z) voxel:
 *  - Supersample the voxel volume (samples^3 sub-voxels).
 *  - For each sub-voxel, project onto the fluence plane, convert to the
 *    source-local frame, reduce to 2D coordinates and look up primary/secondary
 *  *    fluence via the texture-backed fluence maps.
 *  - Average the lookups over samples and apply inverse-square scaling using
 *    the value sampled from the 3D `d_geo_tex` texture.
 *
 * @param[out] fluence_grid    Device output flattened grid (nx*ny*nz) (float*).
 * @param fluence_map_pri      Primary fluence map as a 2D cudaTextureObject_t
 *                             (unnormalized texel coordinates expected).
 * @param fluence_map_sec      Secondary fluence map as a 2D cudaTextureObject_t.
 * @param num_voxels           Device pointer to int[3] = {nx,ny,nz}.
 * @param corner               Device pointer to float[3] world-space grid corner (x,y,z).
 * @param resolution           Device pointer to float[3] voxel sizes (dx,dy,dz).
 * @param d_geo_tex            3D cudaTextureObject_t containing precomputed geometric
 *                             distances (accessed with texel coordinates x+0.5f, ...).
 * @param source_position      Device pointer to float[3] source position (world coords).
 * @param source_v_x           Device pointer to float[3] source basis vector X.
 * @param source_v_y           Device pointer to float[3] source basis vector Y (plane normal).
 * @param source_v_z           Device pointer to float[3] source basis vector Z.
 * @param source_sad           Source-to-axis distance used in rescaling.
 * @param pri_z                Primary fluence map z coord (used in inverse-square).
 * @param sec_z                Secondary fluence map z coord.
 * @param samples              Supersampling factor per axis (1 = no supersampling).
 *
 * @note The fluence map lookup mapping (cm->mm, +280 offset) preserves the
 *       legacy flattened-array indexing semantics; the texture is sampled at
 *       the texel center (ix+0.5f, iy+0.5f) to match integer indexing.
 */
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
    float source_sad,
    float pri_z,
    float sec_z,
    int samples)
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
        float3 offset_f3 = make_float3(0.0f, 0.0f, 0.0f);
        float3 pos_sample_f3 = make_float3(0.0f, 0.0f, 0.0f);
        float3 ray_direction_f3 = make_float3(0.0f, 0.0f, 0.0f);
        float3 pos_plane_f3 = make_float3(0.0f, 0.0f, 0.0f);
        float3 pos_fluence_map_f3 = make_float3(0.0f, 0.0f, 0.0f);
        float2 pos_fluence_map_2d_f2 = make_float2(0.0f, 0.0f);

        // Get voxel corner position
        position_f3.x = corner_f3.x + resolution_f3.x * x;
        position_f3.y = corner_f3.y + resolution_f3.y * y;
        position_f3.z = corner_f3.z + resolution_f3.z * z;

        // Prepare for supersampling loop
        offset_f3.x = resolution_f3.x / samples;
        offset_f3.y = resolution_f3.y / samples;
        offset_f3.z = resolution_f3.z / samples;

        float fluence_pri = 0;
        float fluence_sec = 0;
        for (int ix = 0; ix < samples; ix++) {
            for (int iy = 0; iy < samples; iy++) {
                for (int iz = 0; iz < samples; iz++) {
                    // Position of sample
                    pos_sample_f3.x = position_f3.x + offset_f3.x / 2 + offset_f3.x * ix;
                    pos_sample_f3.y = position_f3.y + offset_f3.y / 2 + offset_f3.y * iy;
                    pos_sample_f3.z = position_f3.z + offset_f3.z / 2 + offset_f3.z * iz;

                    // Determine position on fluence map plane in global coords
                    ray_direction_f3.x = source_position_f3.x - pos_sample_f3.x;
                    ray_direction_f3.y = source_position_f3.y - pos_sample_f3.y;
                    ray_direction_f3.z = source_position_f3.z - pos_sample_f3.z;
                    line_plane_collision_device(&pos_plane_f3, source_position_f3, ray_direction_f3, source_v_y_f3, 1e-6f);

                    // Convert to source coords
                    pos_fluence_map_f3.x = dot3_device(source_v_x_f3, pos_plane_f3);
                    pos_fluence_map_f3.y = dot3_device(source_v_y_f3, pos_plane_f3);
                    pos_fluence_map_f3.z = dot3_device(source_v_z_f3, pos_plane_f3);

                    // Reduce to 2D
                    pos_fluence_map_2d_f2.x = pos_fluence_map_f3.x;
                    pos_fluence_map_2d_f2.y = pos_fluence_map_f3.z;

                    // Accumulate fluence from primary and secondary sources
                    fluence_pri = fluence_pri + fluence_map_lookup_device(pos_fluence_map_2d_f2, fluence_map_pri) / (samples * samples * samples);
                    fluence_sec = fluence_sec + fluence_map_lookup_device(pos_fluence_map_2d_f2, fluence_map_sec) / (samples * samples * samples);
                }
            }
        }
        // Apply inverse square law
        float d = tex3D<float>(d_geo_tex, x, y, z);
        fluence_pri = fluence_pri * ((source_sad - pri_z) / d) * ((source_sad - pri_z) / d);
        fluence_sec = fluence_sec * ((source_sad - sec_z) / d) * ((source_sad - sec_z) / d);

        // Store result
        int idx = x + y * nx + z * nx * ny;
        fluence_grid[idx] = fluence_pri + fluence_sec;
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
 * @param pri_z              Primary fluence map z coord (pri_z used in inverse-square).
 * @param sec_z              Secondary fluence map z coord (sec_z used in inverse-square).
 * @param samples            Supersampling factor per axis (1 = no supersampling).
 *
 * @note The wrapper creates temporary device-side textures and buffers and
 *       destroys them before returning. For repeated calls with the same
 *       fluence maps it is more efficient to create persistent texture
 *       objects and call a variant that accepts texture handles directly.
 */
void map_fluence(pybind11::array_t<float> fluence_grid, pybind11::array_t<float> fluence_map_pri, pybind11::array_t<float> fluence_map_sec, pybind11::array_t<int> num_voxels,
    pybind11::array_t<float> corner, pybind11::array_t<float> resolution, pybind11::array_t<float> d_geo_grid,
    pybind11::array_t<float> source_position, pybind11::array_t<float> source_v_x, pybind11::array_t<float> source_v_y,
    pybind11::array_t<float> source_v_z, float source_sad,
    float pri_z, float sec_z, int samples)
{
    pybind11::buffer_info fluence_grid_info = fluence_grid.request();
    pybind11::buffer_info fluence_map_pri_info = fluence_map_pri.request();
    pybind11::buffer_info fluence_map_sec_info = fluence_map_sec.request();
    pybind11::buffer_info num_voxels_info = num_voxels.request();
    pybind11::buffer_info corner_info = corner.request();
    pybind11::buffer_info resolution_info = resolution.request();
    pybind11::buffer_info d_geo_grid_info = d_geo_grid.request();
    pybind11::buffer_info source_position_info = source_position.request();
    pybind11::buffer_info source_v_x_info = source_v_x.request();
    pybind11::buffer_info source_v_y_info = source_v_y.request();
    pybind11::buffer_info source_v_z_info = source_v_z.request();

    float* fluence_grid_ptr = reinterpret_cast<float*>(fluence_grid_info.ptr);
    float* fluence_map_pri_ptr = reinterpret_cast<float*>(fluence_map_pri_info.ptr);
    float* fluence_map_sec_ptr = reinterpret_cast<float*>(fluence_map_sec_info.ptr);
    int* num_voxels_ptr = reinterpret_cast<int*>(num_voxels_info.ptr);
    float* corner_ptr = reinterpret_cast<float*>(corner_info.ptr);
    float* resolution_ptr = reinterpret_cast<float*>(resolution_info.ptr);
    float* d_geo_grid_ptr = reinterpret_cast<float*>(d_geo_grid_info.ptr);
    float* source_position_ptr = reinterpret_cast<float*>(source_position_info.ptr);
    float* source_v_x_ptr = reinterpret_cast<float*>(source_v_x_info.ptr);
    float* source_v_y_ptr = reinterpret_cast<float*>(source_v_y_info.ptr);
    float* source_v_z_ptr = reinterpret_cast<float*>(source_v_z_info.ptr);

    // Allocate device memory and copy inputs
    float *d_fluence_grid, *d_corner, *d_resolution, *d_source_position, *d_source_v_x_ptr, *d_source_v_y_ptr, *d_source_v_z_ptr;
    int* d_num_voxels;
    cudaMalloc(&d_fluence_grid, fluence_grid_info.size * sizeof(float));
    cudaMalloc(&d_num_voxels, num_voxels_info.size * sizeof(int));
    cudaMalloc(&d_corner, corner_info.size * sizeof(float));
    cudaMalloc(&d_resolution, resolution_info.size * sizeof(float));
    cudaMalloc(&d_source_position, source_position_info.size * sizeof(float));
    cudaMalloc(&d_source_v_x_ptr, source_v_x_info.size * sizeof(float));
    cudaMalloc(&d_source_v_y_ptr, source_v_y_info.size * sizeof(float));
    cudaMalloc(&d_source_v_z_ptr, source_v_z_info.size * sizeof(float));
    cudaMemcpy(d_fluence_grid, fluence_grid_ptr, fluence_grid_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_voxels, num_voxels_ptr, num_voxels_info.size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_corner, corner_ptr, corner_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_resolution, resolution_ptr, resolution_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_position, source_position_ptr, source_position_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_v_x_ptr, source_v_x_ptr, source_v_x_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_v_y_ptr, source_v_y_ptr, source_v_y_info.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_v_z_ptr, source_v_z_ptr, source_v_z_info.size * sizeof(float), cudaMemcpyHostToDevice);

    // Create texture object from host density array (helper in texture_utils.cuh)
    Texture3DHandle d_geo_tex_handle = create_texture3d_from_ptr(d_geo_grid_ptr, num_voxels_ptr[0], num_voxels_ptr[1], num_voxels_ptr[2], cudaFilterModeLinear);
    cudaTextureObject_t d_geo_tex = d_geo_tex_handle.tex;
    Texture2DHandle fluence_map_pri_tex_handle = create_texture2d_from_ptr(fluence_map_pri_ptr, fluence_map_pri_info.shape[0], fluence_map_pri_info.shape[1], cudaFilterModeLinear);
    cudaTextureObject_t fluence_map_pri_tex = fluence_map_pri_tex_handle.tex;
    Texture2DHandle fluence_map_sec_tex_handle = create_texture2d_from_ptr(fluence_map_sec_ptr, fluence_map_sec_info.shape[0], fluence_map_sec_info.shape[1], cudaFilterModeLinear);
    cudaTextureObject_t fluence_map_sec_tex = fluence_map_sec_tex_handle.tex;

    // Launch kernel
    dim3 dimBlock(16, 4, 4);
    dim3 dimGrid((num_voxels_ptr[0] + dimBlock.x - 1) / dimBlock.x,
        (num_voxels_ptr[1] + dimBlock.y - 1) / dimBlock.y,
        (num_voxels_ptr[2] + dimBlock.z - 1) / dimBlock.z);

    // Pass texture objects for the fluence maps and d_geo
    fluence<<<dimGrid, dimBlock>>>(d_fluence_grid, fluence_map_pri_tex, fluence_map_sec_tex, d_num_voxels, d_corner,
        d_resolution, d_geo_tex, d_source_position, d_source_v_x_ptr, d_source_v_y_ptr, d_source_v_z_ptr,
        source_sad, pri_z, sec_z, samples);

    // Copy result back to host
    cudaMemcpy(fluence_grid_ptr, d_fluence_grid,
        fluence_grid_info.size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_fluence_grid);
    cudaFree(d_num_voxels);
    cudaFree(d_corner);
    cudaFree(d_resolution);
    cudaFree(d_source_position);
    cudaFree(d_source_v_x_ptr);
    cudaFree(d_source_v_y_ptr);
    cudaFree(d_source_v_z_ptr);

    // Destroy textures and free CUDA arrays (helpers)
    destroy_texture3d(d_geo_tex_handle);
    destroy_texture2d(fluence_map_pri_tex_handle);
    destroy_texture2d(fluence_map_sec_tex_handle);
}
