__device__ float dot(float *a, float *b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
                
// // 3D vector cross product: out = a x b
// __device__ void cross3(const float *a, const float *b, float *out)
// {
//     out[0] = a[1] * b[2] - a[2] * b[1];
//     out[1] = a[2] * b[0] - a[0] * b[2];
//     out[2] = a[0] * b[1] - a[1] * b[0];
// }

// // 3D vector normalization in-place
// __device__ void normalize3(float *v)
// {
//     float n = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
//     if (n > 0.0f)
//     {
//         v[0] /= n;
//         v[1] /= n;
//         v[2] /= n;
//     }
// }
                
__device__ float* line_plane_collision(float *pos_plane, float *ray_start, float *ray_direction, float *plane_normal, float epsilon)
{
    float ndotu = dot(plane_normal, ray_direction);
    //if (abs(ndotu) < epsilon)
    //{
    //  return false;
    //}

    float w[3];
    w[0] = ray_start[0];
    w[1] = ray_start[1];
    w[2] = ray_start[2];

    float si = -dot(plane_normal, w) / ndotu;
    pos_plane[0] = w[0] + si * ray_direction[0];
    pos_plane[1] = w[1] + si * ray_direction[1];
    pos_plane[2] = w[2] + si * ray_direction[2];

    return pos_plane;
}

// Clearly a target for a 2D texture in future (TODO)
__device__ float block_transmission(float *position, float *block_values)
{
    position[0] = floor(position[0] * 100); // Convert tenth of a mm
    position[1] = floor(position[1] * 100);

    position[0] = position[0] + 2000;
    position[1] = position[1] + 2000;

    // Handle position lying outside the defined blocking area
    for (int i = 0; i < 2; i++)
    {
        if (position[i] < 0 || position[i] > 3999)
        {
            return 0;
        }
    }

    int ix = (int)(position[0]) - 1;
    int iy = (int)(position[1]) - 1;
    // Assuming block_values is a 1D array representing a 4000x4000 grid
    int width = 4000;
    float transmission = block_values[ix + iy * width];
    return transmission;
}

// Clearly a target for a 2D texture in future (TODO)
__device__ float fluence_map_lookup(float *position, float *fluence_map)
{
    float pos_x = floor(position[0] * 10); // Convert to mm
    float pos_y = floor(position[1] * 10); // Convert to mm

    pos_x = pos_x + 280;
    pos_y = pos_y + 280;

    int ix = (int)(pos_x) - 1;
    int iy = (int)(pos_y) - 1;

    // Handle position lying outside the defined blocking area
    if (ix < 0 || ix > 559 || iy < 0 || iy > 559)
    {
        return 0;
    }

    // Assuming fluence_map is a 1D array representing a 560*560 grid
    int width = 560;
    float fluence = fluence_map[ix + iy * width];
    return fluence;
}

__global__ void hit_test(float *blocked_grid, int *num_voxels, float *corner, float *resolution, float *source_position, float *source_v_x, float *source_v_y, float *source_v_z, float *block_values, int samples)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < num_voxels[0] && y < num_voxels[1] && z < num_voxels[2])
    {
        float position[3];
        position[0] = corner[0] + resolution[0] * x;
        position[1] = corner[1] + resolution[1] * y;
        position[2] = corner[2] + resolution[2] * z;

        float offset[3];
        offset[0] = resolution[0] / samples;
        offset[1] = resolution[1] / samples;
        offset[2] = resolution[2] / samples;

        float block_factor = 0;
        for (int ix = 0; ix < samples; ix++)
        {
            for (int iy = 0; iy < samples; iy++)
            {
                for (int iz = 0; iz < samples; iz++)
                {

                    // Position of sample
                    float pos_sample[3];
                    pos_sample[0] = position[0] + offset[0]/2 + offset[0] * ix;
                    pos_sample[1] = position[1] + offset[1]/2 + offset[1] * iy;
                    pos_sample[2] = position[2] + offset[2]/2 + offset[2] * iz;

                    // Determine position on blocking plane in global coords
                    float ray_direction[3];
                    ray_direction[0] = source_position[0] - pos_sample[0];
                    ray_direction[1] = source_position[1] - pos_sample[1];
                    ray_direction[2] = source_position[2] - pos_sample[2];

                    float pos_plane[3];
                    line_plane_collision(pos_plane, source_position, ray_direction, source_v_y, 1e-6);

                    // Convert to source coords
                    float pos_block[3];
                    pos_block[0] = dot(source_v_x, pos_plane);
                    pos_block[1] = dot(source_v_y, pos_plane);
                    pos_block[2] = dot(source_v_z, pos_plane);

                    // Reduce to 2D
                    float pos_block_2d[2];
                    pos_block_2d[0] = pos_block[0];
                    pos_block_2d[1] = pos_block[2];
                    block_factor = block_factor + block_transmission(pos_block_2d, block_values) / (samples*samples*samples);
                }
            }
        }
        blocked_grid[x + y * num_voxels[0] + z * num_voxels[0] * num_voxels[1]] = block_factor;
    }
}

/**
 * @brief Compute off‑axis distance (OAD) per voxel.
 *
 * For each voxel this kernel computes the voxel centre in world coordinates,
 * projects it onto the source fluence plane (using the provided source basis
 * vectors) and writes the radial off‑axis distance sqrt(x^2 + z^2) into
 * oad_grid[idx].
 *
 * @param oad_grid       Device output pointer to flattened grid (nx*ny*nz).
 * @param num_voxels     Device pointer to int[3] containing {nx, ny, nz}.
 * @param corner         Device pointer to float[3] world-space corner coords.
 * @param resolution     Device pointer to float[3] voxel sizes (dx,dy,dz).
 * @param source_position Device pointer to float[3] source position in world coords.
 * @param source_v_x     Device pointer to float[3] source local x axis.
 * @param source_v_y     Device pointer to float[3] source local y axis (plane normal).
 * @param source_v_z     Device pointer to float[3] source local z axis.
 *
 * @note Inputs are contiguous device arrays. This kernel currently uses
 *       make_float3(...) for local work; consider texture/linear interpolation
 *       for large lookup tables (TODO in code).
 *
 * @par Thread mapping
 * Each CUDA thread computes exactly one voxel at (x,y,z) using blockIdx/threadIdx.
 */
__global__ void oad(float *oad_grid, int *num_voxels, float *corner, float *resolution, float *source_position, float *source_v_x, float *source_v_y, float *source_v_z)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = num_voxels[0];
    int ny = num_voxels[1];
    int nz = num_voxels[2];

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

    if (x < nx && y < ny && z < nz)
    {
        // Get voxel position
        position_f3.x = corner_f3.x + resolution_f3.x * (x + 0.5);
        position_f3.y = corner_f3.y + resolution_f3.y * (y + 0.5);
        position_f3.z = corner_f3.z + resolution_f3.z * (z + 0.5);

        // Determine distance/direction to source
        distance_f3.x = source_position_f3.x - position_f3.x;
        distance_f3.y = source_position_f3.y - position_f3.y;
        distance_f3.z = source_position_f3.z - position_f3.z;

        // Project position to iso plane
        line_plane_collision(pos_plane_f3, source_position_f3, distance_f3, source_v_y_f3, 1e-6);

        // Convert to source coords
        pos_source_f3.x = dot(source_v_x_f3, pos_plane_f3);
        pos_source_f3.y = dot(source_v_y_f3, pos_plane_f3);
        pos_source_f3.z = dot(source_v_z_f3, pos_plane_f3);
        int idx = x + y * nx + z * nx * ny;
        oad_grid[idx] = sqrt(pos_source_f3.x * pos_source_f3.x + pos_source_f3.z * pos_source_f3.z);
    }
}           


__global__ void d_geo(float *d_geo, int *num_voxels, float *corner, float *resolution, float *source_position)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < num_voxels[0] && y < num_voxels[1] && z < num_voxels[2])
    {
        // Get voxel position
        float position[3];
        position[0] = corner[0] + resolution[0] * (x + 0.5);
        position[1] = corner[1] + resolution[1] * (y + 0.5);
        position[2] = corner[2] + resolution[2] * (z + 0.5);

        // Determine direction to source
        float ray_direction[3];
        ray_direction[0] = source_position[0] - position[0];
        ray_direction[1] = source_position[1] - position[1];
        ray_direction[2] = source_position[2] - position[2];
        float mag = sqrt(ray_direction[0]*ray_direction[0] + ray_direction[1]*ray_direction[1] + ray_direction[2]*ray_direction[2]);
        int idx = x + y * num_voxels[0] + z * num_voxels[0] * num_voxels[1];
        d_geo[idx] = mag;
    }
}

// Could potentially use a texture for density_grid here (TODO)
__global__ void d_eff(float *d_eff, int *num_voxels, float *corner, float *resolution, float *density_grid, float *source_position)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < num_voxels[0] && y < num_voxels[1] && z < num_voxels[2])
    {

        // Get voxel position
        float position[3];
        position[0] = corner[0] + resolution[0] * (x + 0.5);
        position[1] = corner[1] + resolution[1] * (y + 0.5);
        position[2] = corner[2] + resolution[2] * (z + 0.5);

        // Determine direction to source
        float ray_direction[3];
        ray_direction[0] = source_position[0] - position[0];
        ray_direction[1] = source_position[1] - position[1];
        ray_direction[2] = source_position[2] - position[2];
        float mag = sqrt(ray_direction[0]*ray_direction[0] + ray_direction[1]*ray_direction[1] + ray_direction[2]*ray_direction[2]);
        ray_direction[0] /= mag;
        ray_direction[1] /= mag;
        ray_direction[2] /= mag;

        // Precompute things
        float ds = 0.25 * fmin(fmin(resolution[0], resolution[1]), resolution[2]);
        float total_distance = mag;
        int steps = (int)(total_distance / ds);
        float acc = 0;

        for (int i = 0; i < steps; i++)
        {
            // Move a step toward source
            position[0] += ray_direction[0] * ds;
            position[1] += ray_direction[1] * ds;
            position[2] += ray_direction[2] * ds;

            // Map position → voxel indices
            int ix = (int)((position[0] - corner[0]) / resolution[0]);
            int iy = (int)((position[1] - corner[1]) / resolution[1]);
            int iz = (int)((position[2] - corner[2]) / resolution[2]);
            
            if (ix < 0 || ix >= num_voxels[0] || iy < 0 || iy >= num_voxels[1] || iz < 0 || iz >= num_voxels[2])
            {
                break;  // Ray left grid
            }

            // Accumulate density
            int idx = ix + iy * num_voxels[0] + iz * num_voxels[0] * num_voxels[1];
            acc += density_grid[idx] * ds;
        }

        // Store result
        int idx = x + y * num_voxels[0] + z * num_voxels[0] * num_voxels[1];
        d_eff[idx] = acc;
    }
}



// __global__ void fluence_plane(float *num_bixels, float *corner, float *resolution, float *source_position, float *source_v_y, float pri_s, float pri_x, float pri_y, float pri_z, float sec_s, float sec_x, float sec_y, float sec_z)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x < num_bixels[0] && y < num_bixels[1])
//     {
//         // Get bixel position
//         float position[3];
//         position[0] = corner[0] + resolution[0] * (x + 0.5);
//         position[1] = corner[1] + resolution[1] * (y + 0.5);
//         position[2] = 0.0;

//         // Determine distance/direction to source
//         float distance[3];
//         distance[0] = source_position[0] - position[0];
//         distance[1] = source_position[1] - position[1];
//         distance[2] = source_position[2] - position[2];
//         float mag = sqrt(
//             distance[0] * distance[0] +
//             distance[1] * distance[1] +
//             distance[2] * distance[2]
//         );

//         line_plane_collision(pos_plane, source_position, ray_direction, source_v_y, 1e-6);


//         // Point source
//         float fluence_point = pri_s * pow(source_position[1] / mag, 2);

//         int idx = x + y * num_bixels[0];
//         num_bixels[idx] = fluence_point;
//     }    
// }





// Could use textures at this point for both oad_grid and blocked_grid (TODO)               
__global__ void fluence(float *fluence_grid, float *oad_grid, float *blocked_grid, int *num_voxels, float *corner, float *resolution, float *source_position, float *beam_profile_correction_fs_interp, float beam_profile_correction_dx, float source_sad, float sPri, float sAnn, float zAnn, float rInner, float rOuter, float zExp, float sExp, float kExp)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < num_voxels[0] && y < num_voxels[1] && z < num_voxels[2])
    {
        // Get voxel position
        float position[3];
        position[0] = corner[0] + resolution[0] * (x + 0.5);
        position[1] = corner[1] + resolution[1] * (y + 0.5);
        position[2] = corner[2] + resolution[2] * (z + 0.5);

        // Determine distance/direction to source
        float distance[3];
        distance[0] = source_position[0] - position[0];
        distance[1] = source_position[1] - position[1];
        distance[2] = source_position[2] - position[2];
        float mag = sqrt(
            distance[0] * distance[0] +
            distance[1] * distance[1] +
            distance[2] * distance[2]
        );

        // Point source
        float fluence_point = sPri * pow(source_sad / mag, 2);
        // float fluence_point = sPri;

        int idx = x + y * num_voxels[0] + z * num_voxels[0] * num_voxels[1];
        float oad = oad_grid[idx];

        // Annular source
        float fluence_ann;
        float r_ann = oad * zAnn / source_sad;
        if (r_ann >= rInner && r_ann <= rOuter)
        {
            fluence_ann = sAnn * pow(source_sad - zAnn, 2) / pow(mag - zAnn, 2);
        }
        else
        {
            fluence_ann = 0.0;
        }

        // Exponential source
        if (oad < 2.0) { oad = 2.0; } // Avoid function blowing up near zero
        float r_exp = oad * zExp / source_sad;
        float fluence_exp = sExp / r_exp * exp(-kExp * r_exp) * pow(source_sad - zExp, 2) / pow(mag - zExp, 2);
        
        // Beam profile correction
        int ix = (int)(oad / beam_profile_correction_dx);
        float bpc = beam_profile_correction_fs_interp[ix];
        fluence_grid[idx] = (fluence_point * bpc + fluence_ann + fluence_exp) * blocked_grid[idx];
        // fluence_grid[idx] = fluence_point * blocked_grid[idx];
    }
}


__global__ void fluence_new(float *fluence_grid, float *fluence_map_pri, float *fluence_map_sec, int *num_voxels, float *corner, float *resolution, float *d_geo_grid, float *source_position, float *source_v_x, float *source_v_y, float *source_v_z, float source_sad, float pri_s, float pri_x, float pri_y, float pri_z, float sec_s, float sec_x, float sec_y, float sec_z, int samples)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < num_voxels[0] && y < num_voxels[1] && z < num_voxels[2])
    {
        float position[3];
        position[0] = corner[0] + resolution[0] * x;
        position[1] = corner[1] + resolution[1] * y;
        position[2] = corner[2] + resolution[2] * z;

        float offset[3];
        offset[0] = resolution[0] / samples;
        offset[1] = resolution[1] / samples;
        offset[2] = resolution[2] / samples;

        float fluence_pri = 0;
        float fluence_sec = 0;
        for (int ix = 0; ix < samples; ix++)
        {
            for (int iy = 0; iy < samples; iy++)
            {
                for (int iz = 0; iz < samples; iz++)
                {

                    // Position of sample
                    float pos_sample[3];
                    pos_sample[0] = position[0] + offset[0]/2 + offset[0] * ix;
                    pos_sample[1] = position[1] + offset[1]/2 + offset[1] * iy;
                    pos_sample[2] = position[2] + offset[2]/2 + offset[2] * iz;

                    // Determine position on fluence map plane in global coords
                    float ray_direction[3];
                    ray_direction[0] = source_position[0] - pos_sample[0];
                    ray_direction[1] = source_position[1] - pos_sample[1];
                    ray_direction[2] = source_position[2] - pos_sample[2];

                    float pos_plane[3];
                    line_plane_collision(pos_plane, source_position, ray_direction, source_v_y, 1e-6);
                    // printf("pos_plane: %f, %f, %f\n", pos_plane[0], pos_plane[1], pos_plane[2]);

                    // Convert to source coords
                    float pos_fluence_map[3];
                    pos_fluence_map[0] = dot(source_v_x, pos_plane);
                    pos_fluence_map[1] = dot(source_v_y, pos_plane);
                    pos_fluence_map[2] = dot(source_v_z, pos_plane);
                    // printf("pos_fluence_map: %f, %f, %f\n", pos_fluence_map[0], pos_fluence_map[1], pos_fluence_map[2]);

                    // Reduce to 2D
                    float pos_fluence_map_2d[2];
                    pos_fluence_map_2d[0] = pos_fluence_map[0];
                    pos_fluence_map_2d[1] = pos_fluence_map[2];
                    fluence_pri = fluence_pri + fluence_map_lookup(pos_fluence_map_2d, fluence_map_pri) / (samples*samples*samples);
                    fluence_sec = fluence_sec + fluence_map_lookup(pos_fluence_map_2d, fluence_map_sec) / (samples*samples*samples);
                    // printf("x:%f, y:%f, fluence sample: %f\n", pos_fluence_map_2d[0], pos_fluence_map_2d[1], fluence_map_lookup(pos_fluence_map_2d, fluence_map));
                }
            }
        }
        float d = d_geo_grid[x + y * num_voxels[0] + z * num_voxels[0] * num_voxels[1]];
        fluence_pri = fluence_pri * (source_sad / d) * (source_sad / d);
        fluence_sec = fluence_sec * ((source_sad - 10) / d) * ((source_sad - 10) / d);
        fluence_grid[x + y * num_voxels[0] + z * num_voxels[0] * num_voxels[1]] = fluence_pri + fluence_sec;
    }
}

__global__ void terma(float *terma_grid, float *blocked_grid, float *fluence_grid, float *d_geo_grid, float *d_eff_grid, int *num_voxels, int num_energies, float *energy, float *energy_weights, float *mu_w, float source_sad, float *oad_grid, float *off_axis_softening_fs_interp, float off_axis_softening_dx)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < num_voxels[0] && y < num_voxels[1] && z < num_voxels[2])
    {
        int idx = x + y * num_voxels[0] + z * num_voxels[0] * num_voxels[1];

        float oad = oad_grid[idx];
        int ix = (int)(oad / off_axis_softening_dx);
        float oas = off_axis_softening_fs_interp[ix];
        
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
        for (int i = 0; i < num_energies; i++)
        {
            terma += energy_weights[i] * fluence_grid[idx] * exp(
                -mu_w[i] * (d_eff_grid[idx])
            ) * mu_w[i] * energy[i];
        }
        float no_tilt_descaling = (d_geo_grid[idx] / source_sad) * (d_geo_grid[idx] / source_sad);
        terma_grid[idx] = terma * blocked_grid[idx] * no_tilt_descaling;
    }
}

__global__ void mask(float *mask_grid, float *terma_grid, int *num_voxels, float *corner, float *resolution, float max_distance_cm, float terma_threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int max_voxels_distance = (int)(max_distance_cm / fmin(resolution[0], fmin(resolution[1], resolution[2])));

    if (x < num_voxels[0] && y < num_voxels[1] && z < num_voxels[2])
    {
        int idx = x + y * num_voxels[0] + z * num_voxels[0] * num_voxels[1];

        // We check 8 directions (±x, ±y, ±z)
        for (int ix = -1; ix <= 1; ix = ix + 2)  // Just alternate from negative to positive 1
        {
            for (int iy = -1; iy <= 1; iy = iy + 2)
            {
                for (int iz = -1; iz <= 1; iz = iz + 2)
                {
                    for (int v = 0; v <= max_voxels_distance; v++)
                    {
                        int nx = x + ix * v;
                        int ny = y + iy * v;
                        int nz = z + iz * v;

                        // Ensure neighbor indices are within bounds
                        if (nx >= 0 && nx < num_voxels[0] &&
                            ny >= 0 && ny < num_voxels[1] &&
                            nz >= 0 && nz < num_voxels[2])
                        {
                            int n_idx = nx + ny * num_voxels[0] + nz * num_voxels[0] * num_voxels[1];
                            if (terma_grid[n_idx] >= terma_threshold)
                            {
                                mask_grid[idx] = 1.0f;
                                return;
                            }
                        }
                    }
                }
            }
        }
        mask_grid[idx] = 0.0f;  // No neighbors within threshold
    }
}

__global__ void dose(float *dose_grid, float *resolution, int *num_voxels, float *corner, float *density_grid, float *d_geo_grid, float *terma_grid, float *mask_grid, float *kernel_thetas, float *kernel_phis, float *kernel_omegas, float *kernel, float source_sad, float *source_position, float *source_v_x, float *source_v_y, float *source_v_z, int n_depth_bins, float kernel_depth_res_cm, float max_kernel_depth_cm, float ds_cm)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int nx = num_voxels[0];
    int ny = num_voxels[1];
    int nz = num_voxels[2];
    int idx = x + y * nx + z * nx * ny;

    if (x >= nx || y >= ny || z >= nz)
    {
        return;
    }

    if (mask_grid[idx] == 0.0f)
    {
        // Skip convolution for this voxel
        dose_grid[idx] = terma_grid[idx];
        return;  
    }

    float dx = resolution[0];
    float dy = resolution[1];
    float dz = resolution[2];

    float cx = corner[0] + dx * (x + 0.5);
    float cy = corner[1] + dy * (y + 0.5);
    float cz = corner[2] + dz * (z + 0.5);

    float acc = 0.0f;
    float direction[3];   // direction in kernel-local coordinates
    // float direction_w[3];   // direction rotated to world (tilted kernel)

    // Baking in fixed cone angles for now
    const int n_thetas = 16;
    const int n_phis = 12;

    // Precompute trigonometric values for all thetas and phis
    float theta_rad_arr[n_thetas];
    float phi_rad_arr[n_phis];
    float c_t_arr[n_thetas];
    float s_t_arr[n_thetas];
    float c_p_arr[n_phis];
    float s_p_arr[n_phis];
    for (int it = 0; it < n_thetas; it++)
    {
        theta_rad_arr[it] = kernel_thetas[it] * 3.141592653589793 / 180.0;
        c_t_arr[it] = cos(theta_rad_arr[it]);
        s_t_arr[it] = sin(theta_rad_arr[it]);
    }
    for (int ip = 0; ip < n_phis; ip++)
    {
        phi_rad_arr[ip] = (kernel_phis[ip] - 180.0) * 3.141592653589793 / 180.0;
        c_p_arr[ip] = cos(phi_rad_arr[ip]);
        s_p_arr[ip] = sin(phi_rad_arr[ip]);
    }

    // // Compute local beam axis at this voxel (kernel tilting):
    // // y' (axis) points away from source (downstream direction)
    // float axis[3];
    // axis[0] = cx - source_position[0];
    // axis[1] = cy - source_position[1];
    // axis[2] = cz - source_position[2];
    // normalize3(axis);

    // // Choose a helper vector not parallel to axis, to build an orthonormal frame
    // float tmp[3] = {0.0f, 0.0f, 1.0f};
    // if (fabsf(axis[2]) > 0.99f)
    // {
    //     tmp[0] = 1.0f; tmp[1] = 0.0f; tmp[2] = 0.0f;
    // }
    // // x' = normalize(tmp x axis), z' = axis x x'
    // float xprime[3];
    // float zprime[3];
    // cross3(tmp, axis, xprime);
    // normalize3(xprime);
    // cross3(axis, xprime, zprime);

    for (int it = 0; it < n_thetas; it++)
    {
        for (int ip = 0; ip < n_phis; ip++)
        {
            float s = 0.0f;
            float rad_depth = 0.0f;
            int max_steps = (int)(max_kernel_depth_cm / ds_cm);

            float px = cx;
            float py = cy;
            float pz = cz;

            // Use precomputed trig values to get kernel-local direction
            // Local kernel coordinates are defined with +y as the kernel axis
            direction[0] = c_t_arr[it] * s_p_arr[ip];
            direction[1] = c_p_arr[ip];
            direction[2] = s_t_arr[it] * s_p_arr[ip];
            float mag = sqrt(
                direction[0] * direction[0] +
                direction[1] * direction[1] +
                direction[2] * direction[2]
            );
            direction[0] /= mag;
            direction[1] /= mag;
            direction[2] /= mag;

            // // Rotate kernel-local direction into world using the tilted basis {x', y'=axis, z'}
            // direction_w[0] = direction_k[0] * xprime[0] + direction_k[1] * axis[0] + direction_k[2] * zprime[0];
            // direction_w[1] = direction_k[0] * xprime[1] + direction_k[1] * axis[1] + direction_k[2] * zprime[1];
            // direction_w[2] = direction_k[0] * xprime[2] + direction_k[1] * axis[2] + direction_k[2] * zprime[2];

            for (int step = 0; step < max_steps; step++)
            {
                px += direction[0] * ds_cm;
                py += direction[1] * ds_cm;
                pz += direction[2] * ds_cm;
                s += ds_cm;

                int ix = (int)((px - corner[0]) / dx);
                int iy = (int)((py - corner[1]) / dy);
                int iz = (int)((pz - corner[2]) / dz);
                int idx2 = ix + iy * nx + iz * nx * ny;

                // printf("%d, %d, %d\n", ix, iy, iz);

                if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz)
                {
                    break;  // Ray left grid
                }
                float rho_sample = density_grid[idx2];
                float terma_sample = terma_grid[idx2];
                rad_depth += rho_sample * ds_cm;
                

                int depth_idx = (int)(rad_depth / kernel_depth_res_cm);
                
                if (depth_idx >= n_depth_bins)
                {
                    break;  // Beyond end of kernel
                }
                float kernel_value = kernel[ip * n_depth_bins + depth_idx];
                float vol = kernel_omegas[ip] * ds_cm * s * s; // Volume of sample sector
                acc += terma_sample * kernel_value * vol;
                if (s >= max_kernel_depth_cm)
                {
                    break;
                }
            }
        }
    }
    float no_tilt_rescaling = (source_sad / d_geo_grid[idx]) * (source_sad / d_geo_grid[idx]);
    // float no_tilt_rescaling = 1.0f;
    dose_grid[idx] = acc * no_tilt_rescaling;
}

// Spectral-aware, banked-kernel convolution.
// Uses a precomputed kernel bank indexed by total water-equivalent depth T_eff = d_eff - oas.
// The bank layout is [n_T_bins][n_phis][n_depth_bins] flattened in row-major order.
// We use linear interpolation between neighboring T bins.
__global__ void dose_banked(
    float *dose_grid,
    float *resolution,
    int *num_voxels,
    float *corner,
    float *density_grid,
    float *d_geo_grid,
    float *d_eff_grid,
    float *terma_grid,
    float *mask_grid,
    float *oad_grid,
    float *kernel_thetas,
    float *kernel_phis,
    float *kernel_omegas,
    // Bank
    float *kernel_bank,
    int n_T_bins,
    float T_min,
    float T_step,
    // Off-axis softening LUT
    float *off_axis_softening_fs_interp,
    int off_axis_table_len,
    float off_axis_softening_dx,
    // Geom/scales
    float source_sad,
    float *source_position,
    float *source_v_x,
    float *source_v_y,
    float *source_v_z,
    // Kernel sampling
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

    if (x >= nx || y >= ny || z >= nz)
    {
        return;
    }

    if (mask_grid[idx] == 0.0f)
    {
        dose_grid[idx] = terma_grid[idx];
        return;
    }

    float dx = resolution[0];
    float dy = resolution[1];
    float dz = resolution[2];

    float cx = corner[0] + dx * (x + 0.5f);
    float cy = corner[1] + dy * (y + 0.5f);
    float cz = corner[2] + dz * (z + 0.5f);

    float acc = 0.0f;

    // Fixed angular sampling to match existing path
    const int n_thetas = 16;
    const int n_phis = 12;

    // Precompute trig tables
    float theta_rad_arr[n_thetas];
    float phi_rad_arr[n_phis];
    float c_t_arr[n_thetas];
    float s_t_arr[n_thetas];
    float c_p_arr[n_phis];
    float s_p_arr[n_phis];
    for (int it = 0; it < n_thetas; it++)
    {
        theta_rad_arr[it] = kernel_thetas[it] * 3.141592653589793f / 180.0f;
        c_t_arr[it] = cosf(theta_rad_arr[it]);
        s_t_arr[it] = sinf(theta_rad_arr[it]);
    }
    for (int ip = 0; ip < n_phis; ip++)
    {
        phi_rad_arr[ip] = (kernel_phis[ip] - 180.0f) * 3.141592653589793f / 180.0f;
        c_p_arr[ip] = cosf(phi_rad_arr[ip]);
        s_p_arr[ip] = sinf(phi_rad_arr[ip]);
    }

    for (int it = 0; it < n_thetas; it++)
    {
        for (int ip = 0; ip < n_phis; ip++)
        {
            float s = 0.0f;
            float rad_depth = 0.0f;
            int max_steps = (int)(max_kernel_depth_cm / ds_cm);

            float px = cx;
            float py = cy;
            float pz = cz;

            float dirx = c_t_arr[it] * s_p_arr[ip];
            float diry = c_p_arr[ip];
            float dirz = s_t_arr[it] * s_p_arr[ip];

            for (int step = 0; step < max_steps; step++)
            {
                px += dirx * ds_cm;
                py += diry * ds_cm;
                pz += dirz * ds_cm;
                s += ds_cm;

                int ix = (int)((px - corner[0]) / dx);
                int iy = (int)((py - corner[1]) / dy);
                int iz = (int)((pz - corner[2]) / dz);
                if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz)
                {
                    break;  // Ray left grid
                }
                int idx2 = ix + iy * nx + iz * nx * ny;

                float rho_sample = density_grid[idx2];
                float terma_sample = terma_grid[idx2];
                rad_depth += rho_sample * ds_cm;

                int depth_idx = (int)(rad_depth / kernel_depth_res_cm);
                if (depth_idx >= n_depth_bins)
                {
                    break;  // Beyond end of kernel
                }

                // Total water-equivalent depth at the source voxel for this scatter sample
                float dEff = d_eff_grid[idx2];
                float oad = oad_grid[idx2];
                int lut_ix = (int)(oad / off_axis_softening_dx);
                if (lut_ix < 0) lut_ix = 0;
                if (lut_ix >= off_axis_table_len) lut_ix = off_axis_table_len - 1;
                float oas = off_axis_softening_fs_interp[lut_ix];
                // Assuming oas ~ -T_offaxis (cm water eq.), hence T_eff = d_eff - oas
                float T_eff = dEff - oas;

                // Map T_eff to nearest bank index (no interpolation)
                float u = (T_eff - T_min) / T_step;
                int ib = (int)floorf(u + 0.5f);  // nearest neighbor
                if (ib < 0) ib = 0;
                if (ib >= n_T_bins) ib = n_T_bins - 1;

                int stride_phi_depth = n_depth_bins * n_phis;
                int base = ib * stride_phi_depth + ip * n_depth_bins + depth_idx;
                float kernel_value = kernel_bank[base];

                float vol = kernel_omegas[ip] * ds_cm * s * s; // Volume of sample sector
                acc += terma_sample * kernel_value * vol;
                if (s >= max_kernel_depth_cm)
                {
                    break;
                }
            }
        }
    }

    float no_tilt_rescaling = (source_sad / d_geo_grid[idx]) * (source_sad / d_geo_grid[idx]);
    dose_grid[idx] = acc * no_tilt_rescaling;
}
