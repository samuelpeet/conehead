# %%
import numpy as np
import matplotlib.pyplot as plt
import os; os.environ["NUMBA_ENABLE_CUDASIM"] = "0"; os.environ["NUMBA_DEBUGINFO"] = "0";
import toml
import numba
import math
from numba import cuda
from conehead.kernel import KernelMono
from conehead.phantom import SimplePhantom
from conehead.source import Source
from conehead.dosegrid import DoseGrid
from conehead.block import Block
from conehead.nist import mu_water


# %%
@cuda.jit(device=True)
def cuda_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@cuda.jit(device=True)
def cuda_line_block_plane_collision(pos_plane, ray_start, ray_direction, plane_normal, epsilon):

    plane_point = cuda.local.array(3, numba.float32)
    plane_point[0] = 0
    plane_point[1] = 0
    plane_point[2] = 0

    ndotu = cuda_dot(plane_normal, ray_direction)
    # if abs(ndotu) < epsilon:
    #     raise RuntimeError("no intersection or line is within plane")

    w = cuda.local.array(3, numba.float32)
    w[0] = ray_start[0] - plane_point[0]
    w[1] = ray_start[1] - plane_point[1]
    w[2] = ray_start[2] - plane_point[2]

    si = -cuda_dot(plane_normal, w) / ndotu
    pos_plane[0] = w[0] + si * ray_direction[0] + plane_point[0]
    pos_plane[1] = w[1] + si * ray_direction[1] + plane_point[1]
    pos_plane[2] = w[2] + si * ray_direction[2] + plane_point[2]

    return pos_plane

@cuda.jit(device=True)
def cuda_block_transmission(position, block_values):

        position[0] = math.floor(position[0] * numba.float32(100))  # Convert tenth of a mm
        position[1] = math.floor(position[1] * numba.float32(100))

        position[0] = position[0] + numba.float32(2000)
        position[1] = position[1] + numba.float32(2000)

        # Handle position lying outside the defined blocking area
        for coord in position:
            if coord < 0 or coord > 3999:
                return numba.float32(0)

        transmission = block_values[
            int(position[0])-1,
            int(position[1])-1
        ]
        return transmission

@cuda.jit
def cuda_hit_test(dose_grid_blocked, dose_grid_size, dose_grid_origin, dose_grid_spacing, source_position, source_v_y, source_transform, block_values, samples: int):

    x, y, z = cuda.grid(3)
    if x < dose_grid_size[0] and y < dose_grid_size[1] and z < dose_grid_size[2]:

        position = cuda.local.array(3, numba.float32)
        position[0] = dose_grid_origin[0] + dose_grid_spacing[0] * x
        position[1] = dose_grid_origin[1] + dose_grid_spacing[1] * y
        position[2] = dose_grid_origin[2] + dose_grid_spacing[2] * z

        offset = cuda.local.array(3, numba.float32)
        offset[0] = dose_grid_spacing[0] / samples
        offset[1] = dose_grid_spacing[1] / samples
        offset[2] = dose_grid_spacing[2] / samples

        block_factor: numba.float32 = 0
        for ix in range(samples):
            for iy in range(samples):
                for iz in range(samples):

                    # Position of sample
                    pos_sample = cuda.local.array(3, numba.float32)
                    pos_sample[0] = position[0] - offset[0]/2 + offset[0] * ix
                    pos_sample[1] = position[1] - offset[1]/2 + offset[1] * iy
                    pos_sample[2] = position[2] - offset[2]/2 + offset[2] * iz

                    # Determine position on blocking plane in global coords
                    ray_direction = cuda.local.array(3, numba.float32)
                    ray_direction[0] = source_position[0] - pos_sample[0]
                    ray_direction[1] = source_position[1] - pos_sample[1]
                    ray_direction[2] = source_position[2] - pos_sample[2]

                    pos_plane = cuda.local.array(3, numba.float32)
                    pos_plane = cuda_line_block_plane_collision(pos_plane, source_position, ray_direction, source_v_y, 1e-6)

                    # Convert to source coords
                    pos_block = cuda.local.array(3, numba.float32)
                    pos_block[0] = cuda_dot(source_transform[0, :], pos_plane)
                    pos_block[1] = cuda_dot(source_transform[1, :], pos_plane)
                    pos_block[2] = cuda_dot(source_transform[2, :], pos_plane)

                    # Reduce to 2D
                    pos_block_2d = cuda.local.array(2, numba.float32)
                    pos_block_2d[0] = pos_block[0]
                    pos_block_2d[1] = pos_block[2]

                    block_factor = block_factor + cuda_block_transmission(pos_block_2d, block_values) / samples**3

        dose_grid_blocked[x, y, z] = block_factor


@cuda.jit
def cuda_oad(dose_grid_oad, dose_grid_size, dose_grid_origin, dose_grid_spacing, source_position, source_transform, source_v_y):

    x, y, z = cuda.grid(3)

    if x < dose_grid_size[0] and y < dose_grid_size[1] and z < dose_grid_size[2]:

        # Get voxel position
        position = cuda.local.array(3, numba.float32)
        position[0] = dose_grid_origin[0] + dose_grid_spacing[0] * x
        position[1] = dose_grid_origin[1] + dose_grid_spacing[1] * y
        position[2] = dose_grid_origin[2] + dose_grid_spacing[2] * z

        # Determine distance/direction to source
        distance = cuda.local.array(3, numba.float32)
        distance[0] = source_position[0] - position[0]
        distance[1] = source_position[1] - position[1]
        distance[2] = source_position[2] - position[2]

        # Project position to iso plane
        pos_plane = cuda.local.array(3, numba.float32)
        pos_plane = cuda_line_block_plane_collision(pos_plane, source_position, distance, source_v_y, 1e-6)

        # Convert to source coords
        pos_source = cuda.local.array(3, numba.float32)
        pos_source[0] = cuda_dot(source_transform[0, :], pos_plane)
        pos_source[1] = cuda_dot(source_transform[1, :], pos_plane)
        pos_source[2] = cuda_dot(source_transform[2, :], pos_plane)
        dose_grid_oad[x, y, z] = math.sqrt(pos_source[0] * pos_source[0] + pos_source[2] * pos_source[2])


@cuda.jit
def cuda_d_eff(d_eff, dose_grid_size, dose_grid_origin, dose_grid_spacing, dose_grid_densities, source_position):

    x, y, z = cuda.grid(3)
    
    if x < dose_grid_size[0] and y < dose_grid_size[1] and z < dose_grid_size[2]:

        # Get voxel position
        position = cuda.local.array(3, numba.float32)
        position[0] = dose_grid_origin[0] + dose_grid_spacing[0] * x
        position[1] = dose_grid_origin[1] + dose_grid_spacing[1] * y
        position[2] = dose_grid_origin[2] + dose_grid_spacing[2] * z

        # Determine direction to source
        ray_direction = cuda.local.array(3, numba.float32)
        ray_direction[0] = source_position[0] - position[0]
        ray_direction[1] = source_position[1] - position[1]
        ray_direction[2] = source_position[2] - position[2]
        mag = math.sqrt(ray_direction[0]*ray_direction[0] + ray_direction[1]*ray_direction[1] + ray_direction[2]*ray_direction[2])
        ray_direction[0] /= mag
        ray_direction[1] /= mag
        ray_direction[2] /= mag

        # Precompute things
        ds = 0.25 * min(dose_grid_spacing[0], dose_grid_spacing[1], dose_grid_spacing[2])
        total_distance = mag
        steps = int(total_distance / ds)
        acc = numba.float32(0)

        for i in range(steps):
            # Move a step toward source
            position[0] += ray_direction[0] * ds
            position[1] += ray_direction[1] * ds
            position[2] += ray_direction[2] * ds

            # Map position → voxel indices
            ix = int((position[0] - dose_grid_origin[0]) / dose_grid_spacing[0])
            iy = int((position[1] - dose_grid_origin[1]) / dose_grid_spacing[1])
            iz = int((position[2] - dose_grid_origin[2]) / dose_grid_spacing[2])

            if (ix < 0 or ix >= dose_grid_size[0] or
                iy < 0 or iy >= dose_grid_size[1] or
                iz < 0 or iz >= dose_grid_size[2]):
                break  # Ray left grid

            acc += dose_grid_densities[ix, iy, iz] * ds

        d_eff[x, y, z] = acc


@cuda.jit
def cuda_fluence(dose_grid_fluence, dose_grid_oad, dose_grid_blocked, dose_grid_size, dose_grid_origin, dose_grid_spacing, source_position, beam_profile_correction_fs_interp, beam_profile_correction_dx, source_sad, sPri, zAnn, sAnn, rInner, rOuter, zExp, sExp, kExp):
# def cuda_fluence(dose_grid_fluence, dose_grid_blocked, dose_grid_size, dose_grid_origin, dose_grid_spacing, source_position, source_sad, sPri):
    
    x, y, z = cuda.grid(3)

    if x < dose_grid_size[0] and y < dose_grid_size[1] and z < dose_grid_size[2]:

        # Get voxel position
        position = cuda.local.array(3, numba.float32)
        position[0] = dose_grid_origin[0] + dose_grid_spacing[0]/2 + dose_grid_spacing[0] * x
        position[1] = dose_grid_origin[1] + dose_grid_spacing[1]/2 + dose_grid_spacing[1] * y
        position[2] = dose_grid_origin[2] + dose_grid_spacing[2]/2 + dose_grid_spacing[2] * z

        # Determine distance/direction to source
        distance = cuda.local.array(3, numba.float32)
        distance[0] = source_position[0] - position[0]
        distance[1] = source_position[1] - position[1]
        distance[2] = source_position[2] - position[2]
        mag = math.sqrt(
            distance[0] * distance[0] +
            distance[1] * distance[1] +
            distance[2] * distance[2]
        )
        oad = dose_grid_oad[x, y, z]

        # Point source
        fluence_point = sPri * math.pow(source_sad / mag, 2)

        # Annular source
        r_ann = oad * zAnn / source_sad
        if r_ann >= rInner and r_ann <= rOuter:
            fluence_ann = sAnn * math.pow(source_sad - zAnn, 2) / math.pow(mag - zAnn, 2)
        else:
            fluence_ann = 0.0

        # Exponential source
        oad = numba.float32(2) if oad < 2.0 else oad  # Avoid function blowing up near zero
        r_exp = oad * zExp / source_sad
        fluence_exp = sExp / r_exp * math.exp(-kExp * r_exp) * math.pow(source_sad - zExp, 2) / math.pow(mag - zExp, 2)

        # Beam profile correction
        ix = int(oad / beam_profile_correction_dx)
        bpc = beam_profile_correction_fs_interp[ix]

        dose_grid_fluence[x, y, z] = (fluence_point * bpc + fluence_ann + fluence_exp) * dose_grid_blocked[x, y, z]
        # dose_grid_fluence[x, y, z] = fluence_point * dose_grid_blocked[x, y, z]


@cuda.jit
# def cuda_terma(dose_grid_terma, dose_grid_blocked, dose_grid_fluence, dose_grid_d_eff, dose_grid_size, energy, energy_weights, mu_w, f_soften, f_horn):
def cuda_terma(dose_grid_terma, dose_grid_blocked, dose_grid_fluence, dose_grid_d_eff, dose_grid_size, energy, energy_weights, mu_w, dose_grid_oad, off_axis_softening_fs_interp, off_axis_softening_dx):

    x, y, z = cuda.grid(3)

    if x < dose_grid_size[0] and y < dose_grid_size[1] and z < dose_grid_size[2]:

        oad = dose_grid_oad[x, y, z]
        ix = int(oad / off_axis_softening_dx)
        oas = off_axis_softening_fs_interp[ix]

        terma = numba.float32(0)
        # for i in range(len(energy)):
        #     terma += energy_weights[i] * dose_grid_fluence[x, y, z] * math.exp(
        #         -mu_w[i] * f_soften[x, y, z] * dose_grid_d_eff[x, y, z]
        #     ) * energy[i] * mu_w[i] * f_horn[x, y, z]
        for i in range(len(energy)):
            terma += energy_weights[i] * dose_grid_fluence[x, y, z] * math.exp(
                -mu_w[i] * (dose_grid_d_eff[x, y, z] + oas)
            ) * energy[i] * mu_w[i]
        dose_grid_terma[x, y, z] = terma * dose_grid_blocked[x, y, z]


@cuda.jit
def cuda_dose(dose_grid_dose, dose_grid_spacing, dose_grid_size, dose_grid_origin, dose_grid_densities, dose_grid_terma, kernel_thetas, kernel_phis, kernel, source_transform, n_depth_bins, kernel_depth_res_cm, max_kernel_depth_cm, ds_cm):
    x, y, z = cuda.grid(3)
    nx = dose_grid_size[0]
    ny = dose_grid_size[1]
    nz = dose_grid_size[2]

    if x >= nx or y >= ny or z >= nz:
        return

    dx = dose_grid_spacing[0]
    dy = dose_grid_spacing[1]
    dz = dose_grid_spacing[2]

    cx = dose_grid_origin[0] + (x + 0.5) * dx
    cy = dose_grid_origin[1] + (y + 0.5) * dy
    cz = dose_grid_origin[2] + (z + 0.5) * dz

    dose_acc = numba.float32(0.0)
    direction = cuda.local.array(3, numba.float32)

    n_thetas = kernel_thetas.shape[0]
    n_phis = kernel_phis.shape[0]

    # Precompute trigonometric values for all thetas and phis
    theta_rad_arr = cuda.local.array(16, numba.float32)
    phi_rad_arr = cuda.local.array(16, numba.float32)
    c_t_arr = cuda.local.array(16, numba.float32)
    s_t_arr = cuda.local.array(16, numba.float32)
    c_p_arr = cuda.local.array(16, numba.float32)
    s_p_arr = cuda.local.array(16, numba.float32)

    for i in range(n_thetas):
        theta_rad_arr[i] = kernel_thetas[i] * math.pi / 180.0
        c_t_arr[i] = math.cos(theta_rad_arr[i])
        s_t_arr[i] = math.sin(theta_rad_arr[i])
    for j in range(n_phis):
        phi_rad_arr[j] = (kernel_phis[j] - 180.0) * math.pi / 180.0
        c_p_arr[j] = math.cos(phi_rad_arr[j])
        s_p_arr[j] = math.sin(phi_rad_arr[j])

    for i in range(n_thetas):
        for j in range(n_phis):
            s = numba.float32(0.0)
            rad_depth = numba.float32(0.0)
            max_steps = int(max_kernel_depth_cm / ds_cm)

            px = cx
            py = cy
            pz = cz

            # Use precomputed trig values
            direction[0] = c_t_arr[i] * s_p_arr[j]
            direction[1] = c_p_arr[j]
            direction[2] = s_t_arr[i] * s_p_arr[j]
            N = math.sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            direction[0] /= N
            direction[1] /= N
            direction[2] /= N

            for step_i in range(max_steps):
                px += direction[0] * ds_cm
                py += direction[1] * ds_cm
                pz += direction[2] * ds_cm
                s += ds_cm

                ix = int((px - dose_grid_origin[0]) / dx)
                iy = int((py - dose_grid_origin[1]) / dy)
                iz = int((pz - dose_grid_origin[2]) / dz)

                if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                    break

                rho_sample = dose_grid_densities[ix, iy, iz]
                terma_sample = dose_grid_terma[ix, iy, iz]
                rad_depth += rho_sample * ds_cm

                depth_idx_f = rad_depth / kernel_depth_res_cm
                depth_idx = int(depth_idx_f)
                if depth_idx >= n_depth_bins:
                    break

                k_val = kernel[j, depth_idx]
                dose_acc += k_val * terma_sample

                if s >= max_kernel_depth_cm:
                    break

    dose_grid_dose[x, y, z] = dose_acc
 



# %%
settings = toml.load("settings.toml")
kernels = [
    KernelMono("kernels/0.5MeV/0.5MeV.egslst"),
    KernelMono("kernels/1.0MeV/1.0MeV.egslst"),
    KernelMono("kernels/1.5MeV/1.5MeV.egslst"),
    KernelMono("kernels/2.0MeV/2.0MeV.egslst"),
    KernelMono("kernels/2.5MeV/2.5MeV.egslst"),
    KernelMono("kernels/3.0MeV/3.0MeV.egslst"),
    KernelMono("kernels/3.5MeV/3.5MeV.egslst"),
    KernelMono("kernels/4.0MeV/4.0MeV.egslst"),
    KernelMono("kernels/4.5MeV/4.5MeV.egslst"),
    KernelMono("kernels/5.0MeV/5.0MeV.egslst"),
    KernelMono("kernels/5.5MeV/5.5MeV.egslst"),
    KernelMono("kernels/6.0MeV/6.0MeV.egslst"),
]

kernel = np.zeros_like(kernels[0].kernel)
for i in range(len(settings["energy_spectrum"]["energies"])):
    kernel += kernels[i].kernel * settings["energy_spectrum"]["weights"][i]
kernel = kernel / kernel.sum() # normalise
kernel_radii = kernels[0].radii_centres
kernel_phis = kernels[0].angles
kernel_phis_c = kernels[0].angles_centres
kernel_thetas = np.linspace(0, 360 - (360 / 16), 16, dtype=np.float32) # Baking in number of thetas to 16 for now
kernel = kernel / len(kernel_thetas)  # normalise considering number of thetas

beam_profile_correction_oads = np.array(settings["beam_profile_correction"]["oads"], dtype=np.float32)
beam_profile_correction_fs = np.array(settings["beam_profile_correction"]["factors"], dtype=np.float32)
beam_profile_correction_oads_interp = np.linspace(beam_profile_correction_oads[0], beam_profile_correction_oads[-1], 1001, dtype=np.float32)
beam_profile_correction_fs_interp = np.interp(  # Resample to high res for indexing into later
    beam_profile_correction_oads_interp,
    beam_profile_correction_oads,
    beam_profile_correction_fs,
).astype(np.float32)
beam_profile_correction_dx = beam_profile_correction_oads_interp[1] - beam_profile_correction_oads_interp[0]

off_axis_softening_oads = np.array(settings["off_axis_softening"]["oads"], dtype=np.float32)
off_axis_softening_fs = np.array(settings["off_axis_softening"]["factors"], dtype=np.float32)
off_axis_softening_oads_interp = np.linspace(off_axis_softening_oads[0], off_axis_softening_oads[-1], 1001, dtype=np.float32)
off_axis_softening_fs_interp = np.interp(  # Resample to high res for indexing into later
    off_axis_softening_oads_interp,
    off_axis_softening_oads,
    off_axis_softening_fs,
).astype(np.float32)
off_axis_softening_dx = off_axis_softening_oads_interp[1] - off_axis_softening_oads_interp[0]

phantom = SimplePhantom()
source = Source()
dose_grid = DoseGrid(phantom.num_voxels, phantom.corner, phantom.resolution)
block = Block()
block.set_square(np.float32(10))
dose_grid_densities = phantom.densities

# %%
print("Performing hit-testing of dose grid voxels...")
dose_grid_blocked_device = cuda.to_device(np.zeros(dose_grid.num_voxels, dtype=np.float32))
threadsperblock = (16, 4, 4)
blockspergrid_x = math.ceil(dose_grid_blocked_device.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(dose_grid_blocked_device.shape[1] / threadsperblock[1])
blockspergrid_z = math.ceil(dose_grid_blocked_device.shape[2] / threadsperblock[2])
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
cuda_hit_test[blockspergrid, threadsperblock](
    dose_grid_blocked_device,
    cuda.to_device(dose_grid.num_voxels),
    cuda.to_device(dose_grid.corner),
    cuda.to_device(dose_grid.resolution),
    cuda.to_device(source.position),
    cuda.to_device(source.v_y),
    cuda.to_device(source.transform),
    cuda.to_device(block.block_values),
    settings["calculation"]["fluence_resampling"]
)
dose_grid_blocked = dose_grid_blocked_device.copy_to_host()


# %%
print("Calculating off-axis distances")
dose_grid_oad = np.zeros_like(dose_grid_densities, dtype=np.float32)
dose_grid_oad_device = cuda.to_device(dose_grid_oad)
threadsperblock = (16, 4, 4)
blockspergrid_x = math.ceil(dose_grid_oad.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(dose_grid_oad.shape[1] / threadsperblock[1])
blockspergrid_z = math.ceil(dose_grid_oad.shape[2] / threadsperblock[2])
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
cuda_oad[blockspergrid, threadsperblock](
    dose_grid_oad_device,
    cuda.to_device(dose_grid.num_voxels),
    cuda.to_device(dose_grid.corner),
    cuda.to_device(dose_grid.resolution),
    cuda.to_device(source.position),
    cuda.to_device(source.transform),
    cuda.to_device(source.v_y)
)
dose_grid_oad = dose_grid_oad_device.copy_to_host()


# %%
print("Calculating effective depths...")
dose_grid_d_eff = np.zeros_like(dose_grid_densities, dtype=np.float32)
dose_grid_d_eff_device = cuda.to_device(dose_grid_d_eff)
threadsperblock = (16, 4, 4)
blockspergrid_x = math.ceil(dose_grid_d_eff.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(dose_grid_d_eff.shape[1] / threadsperblock[1])
blockspergrid_z = math.ceil(dose_grid_d_eff.shape[2] / threadsperblock[2])
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
cuda_d_eff[blockspergrid, threadsperblock](
    dose_grid_d_eff_device,
    cuda.to_device(dose_grid.num_voxels),
    cuda.to_device(dose_grid.corner),
    cuda.to_device(dose_grid.resolution),
    cuda.to_device(dose_grid_densities),
    cuda.to_device(source.position),
    
)
dose_grid_d_eff = dose_grid_d_eff_device.copy_to_host()


# %%
print("Calculating photon fluence...")
dose_grid_fluence = np.zeros_like(dose_grid_densities, dtype=np.float32)
dose_grid_fluence_device = cuda.to_device(dose_grid_fluence)
threadsperblock = (16, 4, 4)
blockspergrid_x = math.ceil(dose_grid_fluence.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(dose_grid_fluence.shape[1] / threadsperblock[1])
blockspergrid_z = math.ceil(dose_grid_fluence.shape[2] / threadsperblock[2])
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
cuda_fluence[blockspergrid, threadsperblock](
    dose_grid_fluence_device,
    dose_grid_oad_device,
    dose_grid_blocked_device,
    cuda.to_device(dose_grid.num_voxels),
    cuda.to_device(dose_grid.corner),
    cuda.to_device(dose_grid.resolution),
    cuda.to_device(source.position),
    cuda.to_device(beam_profile_correction_fs_interp),
    beam_profile_correction_dx,
    source.sad,
    np.float32(settings["sources"]["s_pri"]),
    np.float32(settings["sources"]['z_ann']),
    np.float32(settings["sources"]['s_ann']),
    np.float32(settings["sources"]['r_inner']),
    np.float32(settings["sources"]['r_outer']),
    np.float32(settings["sources"]['z_exp']),
    np.float32(settings["sources"]['s_exp']),
    np.float32(settings["sources"]['k_exp'])
)
dose_grid_fluence = dose_grid_fluence_device.copy_to_host()


# %%
print("Calculating TERMA...")
energies = np.array([np.float32(x) for x in settings["energy_spectrum"]["energies"]], dtype=np.float32)
energy_weights = np.array([np.float32(x) for x in settings["energy_spectrum"]["weights"]], dtype=np.float32)
mu_w = mu_water(energies)
dose_grid_terma = np.zeros_like(dose_grid_densities, dtype=np.float32)
dose_grid_terma_device = cuda.to_device(dose_grid_terma)
threadsperblock = (16, 4, 4)
blockspergrid_x = math.ceil(dose_grid_fluence.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(dose_grid_fluence.shape[1] / threadsperblock[1])
blockspergrid_z = math.ceil(dose_grid_fluence.shape[2] / threadsperblock[2])
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
cuda_terma[blockspergrid, threadsperblock](
    dose_grid_terma_device,
    dose_grid_blocked_device,
    dose_grid_fluence_device,
    dose_grid_d_eff_device,
    cuda.to_device(dose_grid.num_voxels),
    cuda.to_device(energies),
    cuda.to_device(energy_weights),
    cuda.to_device(mu_w),
    dose_grid_oad_device,
    cuda.to_device(off_axis_softening_fs_interp),
    off_axis_softening_dx,
)
dose_grid_terma = dose_grid_terma_device.copy_to_host()

# tmp = dose_grid_terma[50, :, 50]
# dose_grid_terma = np.zeros_like(dose_grid_densities, dtype=np.float32)
# dose_grid_terma[50, :, 50] = tmp
# dose_grid_terma[50, 50, 50] = 1e3

# %%
print("Calculating dose...")
dose_grid_dose = np.zeros_like(dose_grid_densities, dtype=np.float32)
dose_grid_dose_device = cuda.to_device(dose_grid_dose)
threadsperblock = (16, 4, 4)
blockspergrid_x = math.ceil(dose_grid_dose.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(dose_grid_dose.shape[1] / threadsperblock[1])
blockspergrid_z = math.ceil(dose_grid_dose.shape[2] / threadsperblock[2])
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
cuda_dose[blockspergrid, threadsperblock](
    dose_grid_dose_device,
    cuda.to_device(dose_grid.resolution),
    cuda.to_device(dose_grid.num_voxels),
    cuda.to_device(dose_grid.corner),
    cuda.to_device(dose_grid_densities),
    cuda.to_device(dose_grid_terma),
    cuda.to_device(kernel_thetas),
    cuda.to_device(kernel_phis_c),
    cuda.to_device(kernel),
    cuda.to_device(source.transform),
    800,     # n depth bins
    0.025,   # float32 (e.g. 0.025)
    20,      # float32 (eg n_depth_bins * depth_res)
    0.025    # float32 (ray march step, e.g. 0.025)
)
dose_grid_dose = dose_grid_dose_device.copy_to_host()
# %%
