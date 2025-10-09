__device__ float dot(float *a, float *b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
                
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

    float transmission = block_values[
        (int)(position[0])-1,
        (int)(position[1])-1
    ];
    return transmission;
}
                
__global__ void hit_test(float *dose_grid_blocked, int *dose_grid_size, float *dose_grid_origin, float *dose_grid_spacing, float *source_position, float *source_v_y, float *source_transform, float *block_values, int samples)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < dose_grid_size[0] && y < dose_grid_size[1] && z < dose_grid_size[2])
    {
        float position[3];
        position[0] = dose_grid_origin[0] + dose_grid_spacing[0] * x;
        position[1] = dose_grid_origin[1] + dose_grid_spacing[1] * y;
        position[2] = dose_grid_origin[2] + dose_grid_spacing[2] * z;

        float offset[3];
        offset[0] = dose_grid_spacing[0] / samples;
        offset[1] = dose_grid_spacing[1] / samples;
        offset[2] = dose_grid_spacing[2] / samples;

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
                    pos_plane = line_plane_collision(pos_plane, source_position, ray_direction, source_v_y, 1e-6);

                    // Convert to source coords
                    float pos_block[3];
                    pos_block[0] = dot(source_transform[0, :], pos_plane);
                    pos_block[1] = dot(source_transform[1, :], pos_plane);
                    pos_block[2] = dot(source_transform[2, :], pos_plane);
                
                    // Reduce to 2D
                    float pos_block_2d[2];
                    pos_block_2d[0] = pos_block[0];
                    pos_block_2d[1] = pos_block[2];
                    block_factor = block_factor + block_transmission(pos_block_2d, block_values) / (samples*samples*samples);
                }
            }
        }
        dose_grid_blocked[x, y, z] = block_factor;
    }
}

__global__ void oad(float *dose_grid_oad, int *dose_grid_size, float *dose_grid_origin, float *dose_grid_spacing, float *source_position, float *source_transform, float *source_v_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < dose_grid_size[0] && y < dose_grid_size[1] && z < dose_grid_size[2])
    {

        // Get voxel position
        float position[3];
        position[0] = dose_grid_origin[0] + dose_grid_spacing[0] * (x + 0.5);
        position[1] = dose_grid_origin[1] + dose_grid_spacing[1] * (y + 0.5);
        position[2] = dose_grid_origin[2] + dose_grid_spacing[2] * (z + 0.5);

        // Determine distance/direction to source
        float distance[3];
        distance[0] = source_position[0] - position[0];
        distance[1] = source_position[1] - position[1];
        distance[2] = source_position[2] - position[2];

        // Project position to iso plane
        float pos_plane[3];
        pos_plane = line_plane_collision(pos_plane, source_position, distance, source_v_y, 1e-6);

        // Convert to source coords
        float pos_source[3];
        pos_source[0] = dot(source_transform[0, :], pos_plane);
        pos_source[1] = dot(source_transform[1, :], pos_plane);
        pos_source[2] = dot(source_transform[2, :], pos_plane);
        dose_grid_oad[x, y, z] = sqrt(pos_source[0] * pos_source[0] + pos_source[2] * pos_source[2]);
    }
}           

// Could potentially use a texture for dose_grid_densities here (TODO)
__global__ void d_eff(float *d_eff, int *dose_grid_size, float *dose_grid_origin, float *dose_grid_spacing, float *dose_grid_densities, float *source_position)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < dose_grid_size[0] && y < dose_grid_size[1] && z < dose_grid_size[2])
    {

        // Get voxel position
        float position[3];
        position[0] = dose_grid_origin[0] + dose_grid_spacing[0] * (x + 0.5);
        position[1] = dose_grid_origin[1] + dose_grid_spacing[1] * (y + 0.5);
        position[2] = dose_grid_origin[2] + dose_grid_spacing[2] * (z + 0.5);

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
        float ds = 0.25 * fmin(fmin(dose_grid_spacing[0], dose_grid_spacing[1]), dose_grid_spacing[2]);
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
            int ix = (int)((position[0] - dose_grid_origin[0]) / dose_grid_spacing[0]);
            int iy = (int)((position[1] - dose_grid_origin[1]) / dose_grid_spacing[1]);
            int iz = (int)((position[2] - dose_grid_origin[2]) / dose_grid_spacing[2]);
            
            if (ix < 0 || ix >= dose_grid_size[0] || iy < 0 || iy >= dose_grid_size[1] || iz < 0 || iz >= dose_grid_size[2])
            {
                break;  // Ray left grid
            }

            // Accumulate density
            acc += dose_grid_densities[ix, iy, iz] * ds;
        }

        // Store result
        d_eff[x, y, z] = acc;
    }
}

// Could use textures at this point for both dose_grid_oad and dose_grid_blocked (TODO)               
__global__ void fluence(float *dose_grid_fluence, float *dose_grid_oad, float *dose_grid_blocked, int *dose_grid_size, float *dose_grid_origin, float *dose_grid_spacing, float *source_position, float *beam_profile_correction_fs_interp, float beam_profile_correction_dx, float source_sad, float sPri, float zAnn, float sAnn, float rInner, float rOuter, float zExp, float sExp, float kExp)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < dose_grid_size[0] && y < dose_grid_size[1] && z < dose_grid_size[2])
    {
        // Get voxel position
        float position[3];
        position[0] = dose_grid_origin[0] + dose_grid_spacing[0] * (x + 0.5);
        position[1] = dose_grid_origin[1] + dose_grid_spacing[1] * (y + 0.5);
        position[2] = dose_grid_origin[2] + dose_grid_spacing[2] * (z + 0.5);

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

        float oad = dose_grid_oad[x, y, z];

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
        dose_grid_fluence[x, y, z] = (fluence_point * bpc + fluence_ann + fluence_exp) * dose_grid_blocked[x, y, z];
    }
}

__global__ void terma(float *dose_grid_terma, float *dose_grid_blocked, float *dose_grid_fluence, float *dose_grid_d_eff, int *dose_grid_size, float *energy, float *energy_weights, float *mu_w, float *dose_grid_oad, float *off_axis_softening_fs_interp, float off_axis_softening_dx)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < dose_grid_size[0] && y < dose_grid_size[1] && z < dose_grid_size[2])
    {

        float oad = dose_grid_oad[x, y, z];
        int ix = (int)(oad / off_axis_softening_dx);
        float oas = off_axis_softening_fs_interp[ix];

        float terma = 0;
        for (int i = 0; i < sizeof(energy)/sizeof(energy[0]); i++)
        {
            terma += energy_weights[i] * dose_grid_fluence[x, y, z] * exp(
                -mu_w[i] * (dose_grid_d_eff[x, y, z] + oas)
            ) * energy[i] * mu_w[i];
        }
        dose_grid_terma[x, y, z] = terma * dose_grid_blocked[x, y, z];
    }
}

__global__ void dose(float *dose_grid_dose, float *dose_grid_spacing, int *dose_grid_size, float *dose_grid_origin, float *dose_grid_densities, float *dose_grid_terma, float *kernel_thetas, float *kernel_phis, float *kernel, float *source_transform, int n_depth_bins, float kernel_depth_res_cm, float max_kernel_depth_cm, float ds_cm)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int nx = dose_grid_size[0];
    int ny = dose_grid_size[1];
    int nz = dose_grid_size[2];

    if (x >= nx || y >= ny || z >= nz)
    {
        return;
    }

    float dx = dose_grid_spacing[0];
    float dy = dose_grid_spacing[1];
    float dz = dose_grid_spacing[2];

    float cx = dose_grid_origin[0] + dx * (x + 0.5);
    float cy = dose_grid_origin[1] + dy * (y + 0.5);
    float cz = dose_grid_origin[2] + dz * (z + 0.5);

    float acc = 0;
    float direction[3];

    // Baking in fixed cone angles for now
    int n_thetas = 16;
    int n_phis = 12;

    # Precompute trigonometric values for all thetas and phis
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
        phi_rad_arr[ip] = kernel_phis[ip] * 3.141592653589793 / 180.0;
        c_p_arr[ip] = cos(phi_rad_arr[ip]);
        s_p_arr[ip] = sin(phi_rad_arr[ip]);
    }
    for (int it = 0; it < n_thetas; it++)
    {
        for (int ip = 0; ip < n_phis; ip++)
        {
            float s - 0.0;
            float rad_depth = 0.0;
            max_steps = (int)(max_kernel_depth_cm / ds_cm);

            float px = cx;
            float py = cy;
            float pz = cz;

            // Use precomputed trig values
            direction[0] = c_t_arr[it] * s_p_arr[ip];
            direction[1] = c_t_arr[it];
            direction[2] = s_t_arr[it] * s_p_arr[ip];
            float N = sqrt(direction[0]*direction[0] + direction[1]*direction[1] + direction[2]*direction[2]);
            direction[0] /= N;
            direction[1] /= N;
            direction[2] /= N;

            for (int step = 0; step < max_steps; step++)
            {
                px += direction[0] * ds_cm;
                py += direction[1] * ds_cm;
                pz += direction[2] * ds_cm;

                int ix = (int)((px - dose_grid_origin[0]) / dx);
                int iy = (int)((py - dose_grid_origin[1]) / dy);
                int iz = (int)((pz - dose_grid_origin[2]) / dz);

                if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz)
                {
                    break;  // Ray left grid
                }
                float rho_sample = dose_grid_densities[ix, iy, iz];
                float terma_sample = dose_grid_terma[ix, iy, iz];
                rad_depth += rho_sample * ds_cm;

                int depth_idx = (int)(rad_depth / kernel_depth_res_cm);
                if (depth_idx >= n_depth_bins)
                {
                    break;  // Beyond end of kernel
                }
                float kernel_value = kernel[ip, depth_idx];
                acc += terma_sample * kernel_value;

                if (s > max_kernel_depth_cm)
                {
                    break;
                }
            }
        }
    }
    dose_grid_dose[x, y, z] = acc;
}
