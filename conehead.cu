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

    int ix = (int)(position[0]) - 1;
    int iy = (int)(position[1]) - 1;
    // Assuming block_values is a 1D array representing a 4000x4000 grid
    int width = 4000;
    float transmission = block_values[ix + iy * width];
    return transmission;
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

__global__ void oad(float *oad_grid, int *num_voxels, float *corner, float *resolution, float *source_position, float *source_v_x, float *source_v_y, float *source_v_z)
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

        // Project position to iso plane
        float pos_plane[3];
        line_plane_collision(pos_plane, source_position, distance, source_v_y, 1e-6);

        // Convert to source coords
        float pos_source[3];
        pos_source[0] = dot(source_v_x, pos_plane);
        pos_source[1] = dot(source_v_y, pos_plane);
        pos_source[2] = dot(source_v_z, pos_plane);
        int idx = x + y * num_voxels[0] + z * num_voxels[0] * num_voxels[1];
        oad_grid[idx] = sqrt(pos_source[0] * pos_source[0] + pos_source[2] * pos_source[2]);
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

// Could use textures at this point for both oad_grid and blocked_grid (TODO)               
__global__ void fluence(float *fluence_grid, float *oad_grid, float *blocked_grid, int *num_voxels, float *corner, float *resolution, float *source_position, float *beam_profile_correction_fs_interp, float beam_profile_correction_dx, float source_sad, float sPri, float zAnn, float sAnn, float rInner, float rOuter, float zExp, float sExp, float kExp)
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
    }
}

__global__ void terma(float *terma_grid, float *blocked_grid, float *fluence_grid, float *d_eff_grid, int *num_voxels, int num_energies, float *energy, float *energy_weights, float *mu_w, float *oad_grid, float *off_axis_softening_fs_interp, float off_axis_softening_dx)
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

        float terma = 0;
        for (int i = 0; i < num_energies; i++)
        {
            terma += energy_weights[i] * fluence_grid[idx] * exp(
                -mu_w[i] * (d_eff_grid[idx] + oas)
            ) * energy[i] * mu_w[i];           
        }
        terma_grid[idx] = terma * blocked_grid[idx];
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
                            // printf("Checking neighbor voxel (%d, %d, %d): terma = %f, threshold = %f\n ", nx, ny, nz, terma_grid[n_idx], terma_threshold);
                            if (terma_grid[n_idx] >= terma_threshold)
                            {
                                // printf("Setting voxel (%d, %d, %d) to 1.0\n", x, y, z);
                                mask_grid[idx] = 1.0f;
                                return;
                            }
                        }
                    }
                }
            }
        }
        // printf("Zero (%d, %d, %d)\n", x, y, z);
        mask_grid[idx] = 0.0f;  // No neighbors within threshold
    }
}

__global__ void dose(float *dose_grid, float *resolution, int *num_voxels, float *corner, float *density_grid, float *terma_grid, float *mask_grid, float *kernel_thetas, float *kernel_phis, float *kernel, float *source_v_x, float *source_v_y, float *source_v_z, int n_depth_bins, float kernel_depth_res_cm, float max_kernel_depth_cm, float ds_cm)
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
    float direction[3];

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

            // Use precomputed trig values
            direction[0] = c_t_arr[it] * s_p_arr[ip];
            direction[1] = c_p_arr[ip];
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
                s += ds_cm;

                int ix = (int)((px - corner[0]) / dx);
                int iy = (int)((py - corner[1]) / dy);
                int iz = (int)((pz - corner[2]) / dz);
                int idx = ix + iy * nx + iz * nx * ny;

                // printf("%d, %d, %d\n", ix, iy, iz);

                if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz)
                {
                    break;  // Ray left grid
                }
                float rho_sample = density_grid[idx];
                float terma_sample = terma_grid[idx];
                rad_depth += rho_sample * ds_cm;
                

                int depth_idx = (int)(rad_depth / kernel_depth_res_cm);
                
                if (depth_idx >= n_depth_bins)
                {
                    break;  // Beyond end of kernel
                }
                float kernel_value = kernel[ip * n_depth_bins + depth_idx];
                // printf("%d, %d, %f\n", ip, depth_idx, kernel_value);
                acc += terma_sample * kernel_value;

                if (s >= max_kernel_depth_cm)
                {
                    break;
                }
            }
        }
    }
    dose_grid[idx] = acc;
}





__global__ void active_dose(float *dose_grid, float *resolution, int *num_voxels, float *corner, float *density_grid, float *terma_grid, float *mask_grid, float *kernel_thetas, float *kernel_phis, float *kernel, float *source_v_x, float *source_v_y, float *source_v_z, int n_depth_bins, float kernel_depth_res_cm, float max_kernel_depth_cm, float ds_cm, int active_dose_interp_skip, float active_dose_interp_terma_threshold, float active_dose_interp_dose_threshold)
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

    if (x % active_dose_interp_skip != 0 || y % active_dose_interp_skip != 0 || z % active_dose_interp_skip != 0)
    {
        // Skip this voxel
        // printf("Skipping voxel (%d, %d, %d)\n", x, y, z);
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
    float direction[3];

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

            // Use precomputed trig values
            direction[0] = c_t_arr[it] * s_p_arr[ip];
            direction[1] = c_p_arr[ip];
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
                s += ds_cm;

                int ix = (int)((px - corner[0]) / dx);
                int iy = (int)((py - corner[1]) / dy);
                int iz = (int)((pz - corner[2]) / dz);
                int idx = ix + iy * nx + iz * nx * ny;

                // printf("%d, %d, %d\n", ix, iy, iz);

                if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz)
                {
                    break;  // Ray left grid
                }
                float rho_sample = density_grid[idx];
                float terma_sample = terma_grid[idx];
                rad_depth += rho_sample * ds_cm;
                

                int depth_idx = (int)(rad_depth / kernel_depth_res_cm);
                
                if (depth_idx >= n_depth_bins)
                {
                    break;  // Beyond end of kernel
                }
                float kernel_value = kernel[ip * n_depth_bins + depth_idx];
                // printf("%d, %d, %f\n", ip, depth_idx, kernel_value);
                acc += terma_sample * kernel_value;

                if (s >= max_kernel_depth_cm)
                {
                    break;
                }
            }
        }
    }
    dose_grid[idx] = acc;
}