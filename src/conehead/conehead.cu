/**
 * @brief Compute the dot product of two 3-component vectors.
 *
 * Returns the scalar product
 * a.x*b.x + a.y*b.y + a.z*b.z.
 *
 * @param a First input vector (float3). Passed by value.
 * @param b Second input vector (float3). Passed by value.
 * @return The scalar dot product (float).
 */
__device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * @brief Safe ray–plane intersection that writes the hit point and reports success.
 *
 * Computes P = ray_start + s * ray_direction such that plane_normal ⋅ P = 0
 * (plane is assumed to pass through the origin of the given coordinate frame).
 *
 * @param[out] out_pos_plane  Pointer to float3 that will be written with the intersection point when true.
 * @param ray_start           Ray origin (float3), passed by value.
 * @param ray_direction       Ray direction (float3). Not required to be normalized.
 * @param plane_normal        Plane normal (float3), expressed in the same coordinate frame.
 * @param epsilon             Threshold to detect parallelism (use a small value like 1e-6f).
 *
 * @return true if a unique intersection exists and out_pos_plane is written;
 *         false if the ray is parallel or nearly parallel to the plane
 *         (i.e. |plane_normal ⋅ ray_direction| < epsilon).
 *
 * @note If the plane does not pass through the origin, subtract a known point on
 *       the plane from ray_start before calling or extend the API to include a plane offset.
 *       Caller must decide how to handle a false result (skip sample, use a default, etc.).
 */
__device__ bool line_plane_collision(float3 *out_pos_plane,
                                     float3 ray_start,
                                     float3 ray_direction,
                                     float3 plane_normal,
                                     float epsilon)
{
    float ndotu = dot(plane_normal, ray_direction);
    if (fabsf(ndotu) < epsilon) {
        // Parallel (or nearly parallel). Caller should skip or handle specially.
        return false;
    }
    float si = -dot(plane_normal, ray_start) / ndotu;
    out_pos_plane->x = ray_start.x + si * ray_direction.x;
    out_pos_plane->y = ray_start.y + si * ray_direction.y;
    out_pos_plane->z = ray_start.z + si * ray_direction.z;
    return true;
}

/**
 * @brief Lookup fluence value from a flattened 2D fluence map.
 *
 * Map a 2D position to integer map indices and return the fluence stored
 * in the flattened row-major map. Behavior and assumptions:
 *  - The fluence map is a 560 × 560 grid.
 *  - Each map element represents 1 mm.
 *  - Input position is expected in centimeters (cm). The function multiplies
 *    position by 10 to convert to millimetres (pos_mm = floor(position * 10)).
 *  - A centering offset of +280 mm is applied to both axes before indexing.
 *  - Out-of-bounds lookups return 0.0f.
 *
 * @param position   2D position (float2) in source-local/world coords (cm).
 * @param fluence_map Flattened float array of length 560*560 (row-major).
 * @return Fluence value at the mapped location, or 0.0f if outside bounds.
 */
__device__ float fluence_map_lookup(float2 position, float *fluence_map)
{
    float pos_x = floor(position.x * 10); // Convert to mm
    float pos_y = floor(position.y * 10); // Convert to mm

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

/**
 * @brief Compute off‑axis distance (OAD) per voxel.
 *
 * For each voxel this kernel computes the voxel centre in world coordinates,
 * projects it onto the source fluence plane (using the provided source basis
 * vectors) and writes the radial off‑axis distance sqrt(x^2 + z^2) into
 * oad_grid[idx].
 *
 * @param[out] oad_grid       Device output pointer to flattened grid (nx*ny*nz).
 * @param num_voxels     Device pointer to int[3] containing {nx, ny, nz}.
 * @param corner         Device pointer to float[3] world-space corner coords.
 * @param resolution     Device pointer to float[3] voxel sizes (dx,dy,dz).
 * @param source_position Device pointer to float[3] source position in world coords.
 * @param source_v_x     Device pointer to float[3] source local x axis.
 * @param source_v_y     Device pointer to float[3] source local y axis (plane normal).
 * @param source_v_z     Device pointer to float[3] source local z axis.
 */
__global__ void oad(float *oad_grid,
                    int *num_voxels,
                    float *corner,
                    float *resolution,
                    float *source_position,
                    float *source_v_x,
                    float *source_v_y,
                    float *source_v_z)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = num_voxels[0];
    int ny = num_voxels[1];
    int nz = num_voxels[2];

    if (x < nx && y < ny && z < nz)
    {
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

        // Get voxel position
        position_f3.x = corner_f3.x + resolution_f3.x * (x + 0.5);
        position_f3.y = corner_f3.y + resolution_f3.y * (y + 0.5);
        position_f3.z = corner_f3.z + resolution_f3.z * (z + 0.5);

        // Determine distance/direction to source
        distance_f3.x = source_position_f3.x - position_f3.x;
        distance_f3.y = source_position_f3.y - position_f3.y;
        distance_f3.z = source_position_f3.z - position_f3.z;

        // Project position to iso plane
        line_plane_collision(&pos_plane_f3, source_position_f3, distance_f3, source_v_y_f3, 1e-6f);

        // Convert to source coords
        pos_source_f3.x = dot(source_v_x_f3, pos_plane_f3);
        pos_source_f3.y = dot(source_v_y_f3, pos_plane_f3);
        pos_source_f3.z = dot(source_v_z_f3, pos_plane_f3);
        int idx = x + y * nx + z * nx * ny;
        oad_grid[idx] = sqrt(pos_source_f3.x * pos_source_f3.x + pos_source_f3.z * pos_source_f3.z);
    }
}           

/**
 * @brief Compute geometric distance from the source to each voxel centre.
 *
 * Each CUDA thread computes the Euclidean distance from the provided source
 * position to the centre of a single voxel and writes that scalar into
 * d_geo_grid at the flattened index (x + y*nx + z*nx*ny).
 *
 * @param[out] d_geo_grid  Device output pointer to flattened grid (nx*ny*nz) where distances are written.
 * @param num_voxels     Device pointer to int[3] containing {nx, ny, nz}.
 * @param corner         Device pointer to float[3] world-space corner coordinates of the grid.
 * @param resolution     Device pointer to float[3] voxel sizes (dx, dy, dz).
 * @param source_position Device pointer to float[3] source position in world coordinates.
 */
__global__ void d_geo(float *d_geo_grid,
                      int *num_voxels,
                      float *corner,
                      float *resolution,
                      float *source_position)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = num_voxels[0];
    int ny = num_voxels[1];
    int nz = num_voxels[2];

    if (x < nx && y < ny && z < nz)
    {
        float3 corner_f3 = make_float3(corner[0], corner[1], corner[2]);
        float3 resolution_f3 = make_float3(resolution[0], resolution[1], resolution[2]);
        float3 source_position_f3 = make_float3(source_position[0], source_position[1], source_position[2]);
        float3 position_f3 = make_float3(0.0f, 0.0f, 0.0f);
        float3 ray_direction_f3 = make_float3(0.0f, 0.0f, 0.0f);

        // Get voxel position
        position_f3.x = corner_f3.x + resolution_f3.x * (x + 0.5);
        position_f3.y = corner_f3.y + resolution_f3.y * (y + 0.5);
        position_f3.z = corner_f3.z + resolution_f3.z * (z + 0.5);

        // Determine direction/distance to source
        ray_direction_f3.x = source_position_f3.x - position_f3.x;
        ray_direction_f3.y = source_position_f3.y - position_f3.y;
        ray_direction_f3.z = source_position_f3.z - position_f3.z;
        float mag = sqrt(ray_direction_f3.x * ray_direction_f3.x + ray_direction_f3.y * ray_direction_f3.y + ray_direction_f3.z * ray_direction_f3.z);
        int idx = x + y * nx + z * nx * ny;
        d_geo_grid[idx] = mag;
    }
}

/**
 * @brief Compute radiological (effective) depth from each voxel toward the source.
 *
 * This kernel implements a simple ray-marching integration from the centre of
 * each voxel toward the source position. For each step along the ray it
 * samples the provided density_grid and accumulates density * ds into an
 * accumulator which is written to d_eff_grid at the voxel's flattened index.
 * The per-step increment ds is computed as:
 *       ds = 0.25 * min(dx, dy, dz)
 * where dx/dy/dz are the voxel resolutions passed in `resolution`.
 *
 * @param[out] d_eff_grid     Device output pointer to flattened grid (nx*ny*nz).
 * @param num_voxels     Device pointer to int[3] containing {nx, ny, nz}.
 * @param corner         Device pointer to float[3] world-space corner coordinates.
 * @param resolution     Device pointer to float[3] voxel sizes (dx, dy, dz).
 * @param density_grid   Device pointer to flattened density grid co-located with voxels.
 * @param source_position Device pointer to float[3] source position in world coordinates.
 */
__global__ void d_eff(float *d_eff_grid,
                      int *num_voxels,
                      float *corner,
                      float *resolution,
                      float *density_grid,
                      float *source_position)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = num_voxels[0];
    int ny = num_voxels[1];
    int nz = num_voxels[2];

    if (x < nx && y < ny && z < nz)
    {
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
        float acc = 0;

        for (int i = 0; i < steps; i++)
        {
            // Move a step toward source
            position_f3.x += ray_direction_f3.x * ds;
            position_f3.y += ray_direction_f3.y * ds;
            position_f3.z += ray_direction_f3.z * ds;

            // Map position → voxel indices
            int ix = (int)((position_f3.x - corner_f3.x) / resolution_f3.x);
            int iy = (int)((position_f3.y - corner_f3.y) / resolution_f3.y);
            int iz = (int)((position_f3.z - corner_f3.z) / resolution_f3.z);

            if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz)
            {
                break;  // Ray left grid
            }

            // Accumulate density
            int idx = ix + iy * nx + iz * nx * ny;
            acc += density_grid[idx] * ds;
        }

        // Store result
        int idx = x + y * nx + z * nx * ny;
        d_eff_grid[idx] = acc;
    }
}

/**
 * @brief Compute fluence per voxel by projecting samples to fluence planes.
 *
 * For each voxel this kernel optionally supersamples the voxel volume (samples^3
 * sub-voxels). For each sample it:
 *  1. Computes the world-space sample position.
 *  2. Forms a ray from the sample toward the source and projects the sample
 *     onto the fluence plane (plane that passes through isocentre; normal = source_v_y).
 *  3. Converts the projected point into the source-local coordinate frame
 *     (using source_v_x, source_v_y, source_v_z) and reduces to a 2D plane
 *     coordinate (x,z).
 *  4. Looks up primary and secondary fluence values via fluence_map_lookup()
 *     and accumulates them.
 *  5. After all samples, applies inverse-square scaling using d_geo (and the
 *     hard-coded pri_z/sec_z offsets) and stores fluence_pri + fluence_sec.
 *
 * @param[out] fluence_grid    Device output flattened grid (nx*ny*nz).
 * @param fluence_map_pri      Device pointer to primary fluence 2D map (flattened).
 * @param fluence_map_sec      Device pointer to secondary fluence 2D map (flattened).
 * @param num_voxels           Device pointer to int[3] = {nx,ny,nz}.
 * @param corner               Device pointer to float[3] world-space grid corner.
 * @param resolution           Device pointer to float[3] voxel sizes (dx,dy,dz).
 * @param d_geo_grid           Device pointer to precomputed geometric distances (nx*ny*nz).
 * @param source_position      Device pointer to float[3] source position (world coords).
 * @param source_v_x/y/z       Device pointer to float[3] source basis vectors (x,y(normal),z).
 * @param source_sad           Source-to-axis distance used in rescaling.
 * @param pri_s, pri_x,y,z     Per-primary-map parameters (map scale/offsets). pri_z is used
 *                             in the inverse-square correction: ((source_sad - pri_z)/d)^2.
 * @param sec_s, sec_x,y,z     Per-secondary-map parameters (map scale/offsets). sec_z likewise used.
 * @param samples              Supersampling factor per axis (1 = no supersampling).
 *
 * @par Supersampling
 * The kernel averages over samples^3 sub-voxels. Each sample contributes
 * fluence_map_lookup(...) / (samples*samples*samples) to the accumulated fluence.
 */
__global__ void fluence(float *fluence_grid,
                        float *fluence_map_pri,
                        float *fluence_map_sec,
                        int *num_voxels,
                        float *corner,
                        float *resolution,
                        float *d_geo_grid,
                        float *source_position,
                        float *source_v_x,
                        float *source_v_y,
                        float *source_v_z,
                        float source_sad,
                        float pri_s, float pri_x, float pri_y, float pri_z,
                        float sec_s, float sec_x, float sec_y, float sec_z,
                        int samples)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = num_voxels[0];
    int ny = num_voxels[1];
    int nz = num_voxels[2];

    if (x < nx && y < ny && z < nz)
    {

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
        for (int ix = 0; ix < samples; ix++)
        {
            for (int iy = 0; iy < samples; iy++)
            {
                for (int iz = 0; iz < samples; iz++)
                {
                    // Position of sample
                    pos_sample_f3.x = position_f3.x + offset_f3.x / 2 + offset_f3.x * ix;
                    pos_sample_f3.y = position_f3.y + offset_f3.y / 2 + offset_f3.y * iy;
                    pos_sample_f3.z = position_f3.z + offset_f3.z / 2 + offset_f3.z * iz;

                    // Determine position on fluence map plane in global coords
                    ray_direction_f3.x = source_position_f3.x - pos_sample_f3.x;
                    ray_direction_f3.y = source_position_f3.y - pos_sample_f3.y;
                    ray_direction_f3.z = source_position_f3.z - pos_sample_f3.z;
                    line_plane_collision(&pos_plane_f3, source_position_f3, ray_direction_f3, source_v_y_f3, 1e-6f);

                    // Convert to source coords
                    pos_fluence_map_f3.x = dot(source_v_x_f3, pos_plane_f3);
                    pos_fluence_map_f3.y = dot(source_v_y_f3, pos_plane_f3);
                    pos_fluence_map_f3.z = dot(source_v_z_f3, pos_plane_f3);

                    // Reduce to 2D
                    pos_fluence_map_2d_f2.x = pos_fluence_map_f3.x;
                    pos_fluence_map_2d_f2.y = pos_fluence_map_f3.z;

                    // Accumulate fluence from primary and secondary sources
                    fluence_pri = fluence_pri + fluence_map_lookup(pos_fluence_map_2d_f2, fluence_map_pri) / (samples*samples*samples);
                    fluence_sec = fluence_sec + fluence_map_lookup(pos_fluence_map_2d_f2, fluence_map_sec) / (samples*samples*samples);
                }
            }
        }
        // Apply inverse square law
        int idx = x + y * nx + z * nx * ny;
        float d = d_geo_grid[idx];
        fluence_pri = fluence_pri * ((source_sad - pri_z) / d) * ((source_sad - pri_z) / d);
        fluence_sec = fluence_sec * ((source_sad - sec_z) / d) * ((source_sad - sec_z) / d);

        // Store result
        fluence_grid[idx] = fluence_pri + fluence_sec;
    }
}

/**
 * @brief Compute TERMA (total energy released per unit mass/volume) at each voxel.
 *
 * For each voxel this kernel computes a spectrally-weighted TERMA by combining
 * the incident fluence with energy-dependent attenuation and energy deposition
 * factors. The current implementation:
 *   terma = sum_i [ energy_weights[i] * fluence * exp(-mu_w[i] * d_eff) * mu_w[i] * energy[i] ]
 *
 * After the spectral sum the kernel applies the "no-tilt" inverse-square
 * descaling:
 *   terma_out = terma * (d_geo / source_sad)^2
 * This removes the inverse-square falloff from TERMA so the tilted kernel step
 * can reapply the geometric scaling during dose convolution (see dose()).
 *
 * @param[out] terma_grid     Device output flattened grid (nx*ny*nz) where TERMA is written.
 * @param fluence_grid        Device pointer to precomputed fluence (nx*ny*nz).
 * @param d_geo_grid          Device pointer to geometric distances (nx*ny*nz).
 * @param d_eff_grid          Device pointer to effective/radiological depths (nx*ny*nz).
 * @param num_voxels          Device pointer to int[3] = {nx,ny,nz}.
 * @param num_energies        Number of spectral energy bins (length of energy arrays).
 * @param energy              Device pointer to per-bin energy values (length num_energies).
 * @param energy_weights      Device pointer to per-bin weights (length num_energies).
 * @param mu_w                Device pointer to per-bin linear attenuation coefficients (length num_energies).
 * @param source_sad          Source-to-axis distance used for no-tilt descaling.
 * @param oad_grid            Device pointer to off-axis distance grid (nx*ny*nz). Present but not used yet.
 * @param off_axis_softening_fs_interp Device pointer to LUT for off-axis softening (unused, TODO).
 * @param off_axis_softening_dx Grid spacing for off-axis softening LUT (unused, TODO).
 *
 * @note Current limitations / TODO
 * - Off-axis softening is not yet applied (there is a commented-out block showing planned use of oad_grid
 *   and off_axis_softening_fs_interp). Implementing that will modify per-bin energy_weights before summation.
 */
__global__ void terma(float *terma_grid,
                      float *fluence_grid,
                      float *d_geo_grid,
                      float *d_eff_grid,
                      int *num_voxels,
                      int num_energies,
                      float *energy,
                      float *energy_weights,
                      float *mu_w,
                      float source_sad,
                      float *oad_grid,
                      float *off_axis_softening_fs_interp,
                      float off_axis_softening_dx)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = num_voxels[0];
    int ny = num_voxels[1];
    int nz = num_voxels[2];

    if (x < nx && y < ny && z < nz)
    {
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
        for (int i = 0; i < num_energies; i++)
        {
            terma += energy_weights[i] * fluence_grid[idx] * exp(
                -mu_w[i] * (d_eff_grid[idx])
            ) * mu_w[i] * energy[i];
        }
        float no_tilt_descaling = (d_geo_grid[idx] / source_sad) * (d_geo_grid[idx] / source_sad);
        terma_grid[idx] = terma * no_tilt_descaling;
    }
}

/**
 * @brief Build a binary computation mask to accelerate dose convolution.
 *
 * For each voxel this kernel searches outward along the eight diagonal
 * directions (combinations of ±x, ±y, ±z) up to a maximum physical distance
 * (max_distance_cm). If any voxel encountered (including the voxel itself at
 * distance 0) has terma >= terma_threshold the mask for the central voxel is
 * set to 1.0f; otherwise it is set to 0.0f. dose() uses this mask to skip
 * expensive kernel convolution for voxels with negligible nearby TERMA.
 *
 * @param[out] mask_grid         Device output flattened grid (nx*ny*nz) of 0/1 flags.
 * @param terma_grid             Device pointer to flattened TERMA grid (nx*ny*nz).
 * @param num_voxels             Device pointer to int[3] = {nx,ny,nz}.
 * @param resolution             Device pointer to float[3] voxel sizes (dx,dy,dz) in same units as max_distance_cm.
 * @param max_distance_cm        Physical search radius (cm) used to determine neighborhood.
 * @param terma_threshold        TERMA threshold; any neighbor with terma >= this marks the voxel as active.
 */
__global__ void mask(float *mask_grid, 
                     float *terma_grid,
                     int *num_voxels,
                     float *resolution,
                     float max_distance_cm,
                     float terma_threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = num_voxels[0];
    int ny = num_voxels[1];
    int nz = num_voxels[2];

    int max_voxels_distance = (int)(max_distance_cm / fmin(resolution[0], fmin(resolution[1], resolution[2])));

    if (x < nx && y < ny && z < nz)
    {
        int idx = x + y * nx + z * nx * ny;

        // We check 8 directions (±x, ±y, ±z)
        for (int ix = -1; ix <= 1; ix = ix + 2)  // Just alternate from negative to positive 1
        {
            for (int iy = -1; iy <= 1; iy = iy + 2)
            {
                for (int iz = -1; iz <= 1; iz = iz + 2)
                {
                    for (int v = 0; v <= max_voxels_distance; v++)
                    {
                        int cx = x + ix * v;
                        int cy = y + iy * v;
                        int cz = z + iz * v;

                        // Ensure neighbor indices are within bounds
                        if (cx >= 0 && cx < nx &&
                            cy >= 0 && cy < ny &&
                            cz >= 0 && cz < nz)
                        {
                            int n_idx = cx + cy * nx + cz * nx * ny;
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

/**
 * @brief Compute dose by collapsed‑cone gather convolution of TERMA.
 *
 * Performs a gather-style collapsed-cone convolution: for each target voxel
 * (one CUDA thread per voxel) the kernel marches rays along a fixed set of
 * cone directions (predefined theta/phi sampling) and accumulates contributions
 * from TERMA samples encountered along each ray using the precomputed
 * angular-depth kernel. The final per-voxel result is rescaled using the
 * second step of the "no-tilt" approximation (reapply inverse-square).
 *
 * High-level behaviour
 * - One thread → one voxel at (x,y,z); flattened index idx = x + y*nx + z*nx*ny.
 * - If mask_grid[idx] == 0.0f the convolution is skipped and dose_grid[idx] = terma_grid[idx].
 * - For active voxels the kernel:
 *     * Precomputes trig tables from kernel_thetas/kernel_phis (implementation uses n_thetas=16, n_phis=12).
 *     * For each cone direction (it, ip) marches along the ray in steps of ds_cm up to max_kernel_depth_cm.
 *     * At each step it accumulates radiological depth (rad_depth) from density_grid and looks up the
 *       kernel value using depth_idx = int(rad_depth / kernel_depth_res_cm).
 *     * The contribution of a sample is: terma_grid[idx2] * kernel_value * vol
 *       where vol = kernel_omegas[ip] * ds_cm * s * s  (s = distance along ray from voxel centre).
 * - Early exit conditions:
 *     * Ray leaves the grid (index out-of-bounds).
 *     * depth_idx >= n_depth_bins (beyond kernel support).
 *     * s >= max_kernel_depth_cm.
 * - After angular summation the no-tilt re-scaling is applied:
 *       dose_grid[idx] = accumulated * (source_sad / d_geo_grid[idx])^2
 *   (this complements terma()'s descaling step).
 *
 * Important implementation details & assumptions
 * - The method is "gather": samples read terma at sample voxels and contribute to the central target voxel.
 *   This avoids atomic writes but reads many terma values.
 * - Angular sampling and kernel layout:
 *     * Theta/Phi counts are hard-coded in the kernel (n_thetas=16, n_phis=12).
 *     * kernel is stored as flattened [phi][depth] blocks and indexed in code as:
 *         kernel_value = kernel[ip * n_depth_bins + depth_idx];
 *     * kernel_omegas[ip] contains the per-phi solid-angle weight used to compute sample volume.
 * - Memory layout:
 *     * All 3D grids use flattened row-major indexing: idx = x + y*nx + z*nx*ny.
 *
 * Edge cases, robustness & TODOs
 * - The kernel assumes the cone directions are already aligned with the central axis and the source
 *   position used when terma was calculated. If the source is rotated (e.g., gantry 90°) the cone
 *   directions must be rotated accordingly — current implementation does not do this (TODO).
 *
 * @param[out] dose_grid         Device output flattened dose grid (nx*ny*nz).
 * @param resolution             Device pointer to float[3] voxel sizes (dx,dy,dz).
 * @param num_voxels             Device pointer to int[3] = {nx,ny,nz}.
 * @param corner                 Device pointer to float[3] world-space grid corner.
 * @param density_grid           Device pointer to flattened density grid (nx*ny*nz).
 * @param d_geo_grid             Device pointer to geometric distances (nx*ny*nz).
 * @param terma_grid             Device pointer to flattened TERMA grid (nx*ny*nz) (descaled by terma()).
 * @param mask_grid              Device pointer to flattened mask (0.0/1.0) used to skip work.
 * @param kernel_thetas          Array of theta angles (degrees), length n_thetas (16).
 * @param kernel_phis            Array of phi angles (degrees), length n_phis (12).
 * @param kernel_omegas          Array of per-phi solid-angle weights, length n_phis.
 * @param kernel                 Flattened kernel values sized n_phis * n_depth_bins, layout [phi][depth].
 * @param source_sad             Source-to-axis distance used for no-tilt rescaling.
 * @param source_position        Device pointer to float[3] (not currently used for kernel alignment).
 * @param source_v_x/y/z         Device pointers to float[3] source basis vectors (present for context).
 * @param n_depth_bins           Number of depth samples per kernel (kernel depth axis length).
 * @param kernel_depth_res_cm    Depth resolution of kernel bins (cm).
 * @param max_kernel_depth_cm    Maximum kernel depth to march (cm).
 * @param ds_cm                  Ray-marching step size (cm).
 */
__global__ void dose(float *dose_grid, 
                     float *resolution,
                     int *num_voxels,
                     float *corner,
                     float *density_grid,
                     float *d_geo_grid,
                     float *terma_grid,
                     float *mask_grid,
                     float *kernel_thetas,
                     float *kernel_phis,
                     float *kernel_omegas,
                     float *kernel,
                     float source_sad,
                     float *source_position,
                     float *source_v_x,
                     float *source_v_y,
                     float *source_v_z,
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
        corner_f3.z + resolution_f3.z * (z + 0.5)
    );

    // Precompute trigonometric values for all thetas and phis
    const int n_thetas = 16;
    const int n_phis = 12;
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

    // For use in the ray marching loop
    float3 direction_f3 = make_float3(0.0f, 0.0f, 0.0f);
    float3 position_f3 = make_float3(0.0f, 0.0f, 0.0f);
    float dose_acc = 0.0f;

    for (int it = 0; it < n_thetas; it++)
    {
        for (int ip = 0; ip < n_phis; ip++)
        {
            float s = 0.0f;
            float rad_depth = 0.0f;
            int max_steps = (int)(max_kernel_depth_cm / ds_cm);

            // Initialize ray position at voxel centre
            position_f3.x = centre_f3.x;
            position_f3.y = centre_f3.y;
            position_f3.z = centre_f3.z;

            // Use precomputed trig values to get ray direction
            direction_f3.x = c_t_arr[it] * s_p_arr[ip];
            direction_f3.y = c_p_arr[ip];
            direction_f3.z = s_t_arr[it] * s_p_arr[ip];
            float mag = sqrt(
                direction_f3.x * direction_f3.x +
                direction_f3.y * direction_f3.y +
                direction_f3.z * direction_f3.z
            );
            direction_f3.x /= mag;
            direction_f3.y /= mag;
            direction_f3.z /= mag;

            // We have direction and starting position; time to march along ray
            for (int step = 0; step < max_steps; step++)
            {
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
                if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz)
                {
                    break;  // Ray left grid
                }
                // Accumulate radiological depth
                rad_depth += density_grid[idx2] * ds_cm;

                // Lookup kernel value corresponding to this radiological depth
                int depth_idx = (int)(rad_depth / kernel_depth_res_cm);
                if (depth_idx >= n_depth_bins)
                {
                    break;  // Beyond end of kernel
                }
                float kernel_value = kernel[ip * n_depth_bins + depth_idx];
                float vol = kernel_omegas[ip] * ds_cm * s * s; // Volume of sample sector
                dose_acc += terma_grid[idx2] * kernel_value * vol;
                if (s >= max_kernel_depth_cm)
                {
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

// // Spectral-aware, banked-kernel convolution.
// // Uses a precomputed kernel bank indexed by total water-equivalent depth T_eff = d_eff - oas.
// // The bank layout is [n_T_bins][n_phis][n_depth_bins] flattened in row-major order.
// // We use linear interpolation between neighboring T bins.
// __global__ void dose_banked(
//     float *dose_grid,
//     float *resolution,
//     int *num_voxels,
//     float *corner,
//     float *density_grid,
//     float *d_geo_grid,
//     float *d_eff_grid,
//     float *terma_grid,
//     float *mask_grid,
//     float *oad_grid,
//     float *kernel_thetas,
//     float *kernel_phis,
//     float *kernel_omegas,
//     // Bank
//     float *kernel_bank,
//     int n_T_bins,
//     float T_min,
//     float T_step,
//     // Off-axis softening LUT
//     float *off_axis_softening_fs_interp,
//     int off_axis_table_len,
//     float off_axis_softening_dx,
//     // Geom/scales
//     float source_sad,
//     float *source_position,
//     float *source_v_x,
//     float *source_v_y,
//     float *source_v_z,
//     // Kernel sampling
//     int n_depth_bins,
//     float kernel_depth_res_cm,
//     float max_kernel_depth_cm,
//     float ds_cm)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int z = blockIdx.z * blockDim.z + threadIdx.z;
//     int nx = num_voxels[0];
//     int ny = num_voxels[1];
//     int nz = num_voxels[2];
//     int idx = x + y * nx + z * nx * ny;

//     if (x >= nx || y >= ny || z >= nz)
//     {
//         return;
//     }

//     if (mask_grid[idx] == 0.0f)
//     {
//         dose_grid[idx] = terma_grid[idx];
//         return;
//     }

//     float3 res_f3 = make_float3(resolution[0], resolution[1], resolution[2]);
//     float3 corner_f3 = make_float3(corner[0], corner[1], corner[2]);
//     float3 centre_f3 = make_float3(
//         corner_f3.x + res_f3.x * (x + 0.5f),
//         corner_f3.y + res_f3.y * (y + 0.5f),
//         corner_f3.z + res_f3.z * (z + 0.5f)
//     );

//     float acc = 0.0f;

//     // Fixed angular sampling to match existing path
//     const int n_thetas = 16;
//     const int n_phis = 12;

//     // Precompute trig tables
//     float theta_rad_arr[n_thetas];
//     float phi_rad_arr[n_phis];
//     float c_t_arr[n_thetas];
//     float s_t_arr[n_thetas];
//     float c_p_arr[n_phis];
//     float s_p_arr[n_phis];
//     for (int it = 0; it < n_thetas; it++)
//     {
//         theta_rad_arr[it] = kernel_thetas[it] * 3.141592653589793f / 180.0f;
//         c_t_arr[it] = cosf(theta_rad_arr[it]);
//         s_t_arr[it] = sinf(theta_rad_arr[it]);
//     }
//     for (int ip = 0; ip < n_phis; ip++)
//     {
//         phi_rad_arr[ip] = (kernel_phis[ip] - 180.0f) * 3.141592653589793f / 180.0f;
//         c_p_arr[ip] = cosf(phi_rad_arr[ip]);
//         s_p_arr[ip] = sinf(phi_rad_arr[ip]);
//     }

//     for (int it = 0; it < n_thetas; it++)
//     {
//         for (int ip = 0; ip < n_phis; ip++)
//         {
//             float s = 0.0f;
//             float rad_depth = 0.0f;
//             int max_steps = (int)(max_kernel_depth_cm / ds_cm);

//             float px = centre_f3.x;
//             float py = centre_f3.y;
//             float pz = centre_f3.z;

//             float dirx = c_t_arr[it] * s_p_arr[ip];
//             float diry = c_p_arr[ip];
//             float dirz = s_t_arr[it] * s_p_arr[ip];

//             for (int step = 0; step < max_steps; step++)
//             {
//                 px += dirx * ds_cm;
//                 py += diry * ds_cm;
//                 pz += dirz * ds_cm;
//                 s += ds_cm;

//                 int ix = (int)((px - corner_f3.x) / res_f3.x);
//                 int iy = (int)((py - corner_f3.y) / res_f3.y);
//                 int iz = (int)((pz - corner_f3.z) / res_f3.z);
//                 if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz)
//                 {
//                     break;  // Ray left grid
//                 }
//                 int idx2 = ix + iy * nx + iz * nx * ny;

//                 float rho_sample = density_grid[idx2];
//                 float terma_sample = terma_grid[idx2];
//                 rad_depth += rho_sample * ds_cm;

//                 int depth_idx = (int)(rad_depth / kernel_depth_res_cm);
//                 if (depth_idx >= n_depth_bins)
//                 {
//                     break;  // Beyond end of kernel
//                 }

//                 // Total water-equivalent depth at the source voxel for this scatter sample
//                 float dEff = d_eff_grid[idx2];
//                 float oad = oad_grid[idx2];
//                 int lut_ix = (int)(oad / off_axis_softening_dx);
//                 if (lut_ix < 0) lut_ix = 0;
//                 if (lut_ix >= off_axis_table_len) lut_ix = off_axis_table_len - 1;
//                 float oas = off_axis_softening_fs_interp[lut_ix];
//                 // Assuming oas ~ -T_offaxis (cm water eq.), hence T_eff = d_eff - oas
//                 float T_eff = dEff - oas;

//                 // Map T_eff to nearest bank index (no interpolation)
//                 float u = (T_eff - T_min) / T_step;
//                 int ib = (int)floorf(u + 0.5f);  // nearest neighbor
//                 if (ib < 0) ib = 0;
//                 if (ib >= n_T_bins) ib = n_T_bins - 1;

//                 int stride_phi_depth = n_depth_bins * n_phis;
//                 int base = ib * stride_phi_depth + ip * n_depth_bins + depth_idx;
//                 float kernel_value = kernel_bank[base];

//                 float vol = kernel_omegas[ip] * ds_cm * s * s; // Volume of sample sector
//                 acc += terma_sample * kernel_value * vol;
//                 if (s >= max_kernel_depth_cm)
//                 {
//                     break;
//                 }
//             }
//         }
//     }

//     float no_tilt_rescaling = (source_sad / d_geo_grid[idx]) * (source_sad / d_geo_grid[idx]);
//     dose_grid[idx] = acc * no_tilt_rescaling;
// }
