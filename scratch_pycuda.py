# %%
import numpy as np
import matplotlib.pyplot as plt
import toml
import math
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from conehead.kernel import KernelMono
from conehead.phantom import SimplePhantom
from conehead.source import Source
from conehead.dosegrid import DoseGrid
from conehead.block import Block
from conehead.nist import mu_water


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

kernel = np.zeros_like(kernels[0].kernel, dtype=np.float32)
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
grid = DoseGrid(phantom.num_voxels, phantom.corner, phantom.resolution)
block = Block()
block.set_square(np.float32(10))

energies = np.array([np.float32(x) for x in settings["energy_spectrum"]["energies"]], dtype=np.float32)
energy_weights = np.array([np.float32(x) for x in settings["energy_spectrum"]["weights"]], dtype=np.float32)
mu_w = mu_water(energies)


# %%
# Create python data arrays
density_grid = phantom.densities
blocked_grid = np.ones(density_grid.shape, dtype=np.float32)
oad_grid = np.zeros(density_grid.shape, dtype=np.float32)
d_eff_grid = np.zeros(density_grid.shape, dtype=np.float32)
fluence_grid = np.zeros(density_grid.shape, dtype=np.float32)
terma_grid = np.zeros(density_grid.shape, dtype=np.float32)
mask_grid = np.zeros(density_grid.shape, dtype=np.float32)
dose_grid = np.zeros(density_grid.shape, dtype=np.float32)

# Allocate GPU memory
density_grid_gpu = cuda.mem_alloc(density_grid.nbytes)
blocked_grid_gpu = cuda.mem_alloc(density_grid.nbytes)
oad_grid_gpu = cuda.mem_alloc(density_grid.nbytes)
d_eff_grid_gpu = cuda.mem_alloc(density_grid.nbytes)
fluence_grid_gpu = cuda.mem_alloc(density_grid.nbytes)
terma_grid_gpu = cuda.mem_alloc(density_grid.nbytes)
mask_grid_gpu = cuda.mem_alloc(density_grid.nbytes)
dose_grid_gpu = cuda.mem_alloc(density_grid.nbytes)
num_voxels_gpu = cuda.mem_alloc(grid.num_voxels.nbytes)
corner_gpu = cuda.mem_alloc(grid.corner.nbytes)
resolution_gpu = cuda.mem_alloc(grid.resolution.nbytes)
source_position_gpu = cuda.mem_alloc(source.position.nbytes)
source_v_x_gpu = cuda.mem_alloc(source.v_x.nbytes)
source_v_y_gpu = cuda.mem_alloc(source.v_y.nbytes)
source_v_z_gpu = cuda.mem_alloc(source.v_z.nbytes)
block_values_gpu = cuda.mem_alloc(block.block_values.nbytes)
beam_profile_correction_fs_interp_gpu = cuda.mem_alloc(beam_profile_correction_fs_interp.nbytes)
energies_gpu = cuda.mem_alloc(energies.nbytes)
energy_weights_gpu = cuda.mem_alloc(energy_weights.nbytes)
mu_w_gpu = cuda.mem_alloc(mu_w.nbytes)
off_axis_softening_fs_interp_gpu = cuda.mem_alloc(off_axis_softening_fs_interp.nbytes)
kernel_thetas_gpu = cuda.mem_alloc(kernel_thetas.nbytes)
kernel_phis_c_gpu = cuda.mem_alloc(kernel_phis_c.nbytes)
kernel_gpu = cuda.mem_alloc(kernel.nbytes)


# Compile the CUDA kernel
cuda_code = open("conehead.cu").read()
mod = SourceModule(cuda_code)

# Extract kernel functions
hit_test = mod.get_function("hit_test")
oad = mod.get_function("oad")
d_eff = mod.get_function("d_eff")
fluence = mod.get_function("fluence")
terma = mod.get_function("terma")
mask = mod.get_function("mask")
dose = mod.get_function("dose")
active_dose = mod.get_function("active_dose")

# Define grid/block sizes
threadsperblock = (16, 4, 4)
blockspergrid_x = math.ceil(density_grid.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(density_grid.shape[1] / threadsperblock[1])
blockspergrid_z = math.ceil(density_grid.shape[2] / threadsperblock[2])
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

# %%
print("Performing hit-testing of dose grid voxels...")
cuda.memcpy_htod(blocked_grid_gpu, blocked_grid)
cuda.memcpy_htod(num_voxels_gpu, grid.num_voxels)
cuda.memcpy_htod(corner_gpu, grid.corner)
cuda.memcpy_htod(resolution_gpu, grid.resolution)
cuda.memcpy_htod(source_position_gpu, source.position)
cuda.memcpy_htod(source_v_x_gpu, source.v_x)
cuda.memcpy_htod(source_v_y_gpu, source.v_y)
cuda.memcpy_htod(source_v_z_gpu, source.v_z)
cuda.memcpy_htod(block_values_gpu, block.block_values)
hit_test(
    blocked_grid_gpu,
    num_voxels_gpu,
    corner_gpu,
    resolution_gpu,
    source_position_gpu,
    source_v_x_gpu,
    source_v_y_gpu,
    source_v_z_gpu,
    block_values_gpu,
    np.int32(settings["calculation"]["fluence_resampling"]),
    block=threadsperblock,
    grid=blockspergrid
)
cuda.memcpy_dtoh(blocked_grid, blocked_grid_gpu)

# %%
print("Calculating off-axis distances")
cuda.memcpy_htod(oad_grid_gpu, oad_grid)
cuda.memcpy_htod(num_voxels_gpu, grid.num_voxels)
cuda.memcpy_htod(corner_gpu, grid.corner)
cuda.memcpy_htod(resolution_gpu, grid.resolution)
cuda.memcpy_htod(source_position_gpu, source.position)
cuda.memcpy_htod(source_v_x_gpu, source.v_x)
cuda.memcpy_htod(source_v_y_gpu, source.v_y)
cuda.memcpy_htod(source_v_z_gpu, source.v_z)
oad(
    oad_grid_gpu,
    num_voxels_gpu,
    corner_gpu,
    resolution_gpu,
    source_position_gpu,
    source_v_x_gpu,
    source_v_y_gpu,
    source_v_z_gpu,
    block=threadsperblock,
    grid=blockspergrid
)
cuda.memcpy_dtoh(oad_grid, oad_grid_gpu)

# %%
print("Calculating effective depths...")
cuda.memcpy_htod(d_eff_grid_gpu, d_eff_grid)
cuda.memcpy_htod(num_voxels_gpu, grid.num_voxels)
cuda.memcpy_htod(corner_gpu, grid.corner)
cuda.memcpy_htod(resolution_gpu, grid.resolution)
cuda.memcpy_htod(density_grid_gpu, density_grid)
cuda.memcpy_htod(source_position_gpu, source.position)
# start_evt = cuda.Event()
# end_evt = cuda.Event()
# start_evt.record()
d_eff(
    d_eff_grid_gpu,
    num_voxels_gpu,
    corner_gpu,
    resolution_gpu,
    density_grid_gpu,
    source_position_gpu,
    block=threadsperblock,
    grid=blockspergrid,
)
# end_evt.record()
# end_evt.synchronize()
# elapsed_time_ms = start_evt.time_till(end_evt)
# print(f"GPU operation took {elapsed_time_ms:.3f} ms")
cuda.memcpy_dtoh(d_eff_grid, d_eff_grid_gpu)

# %%
print("Calculating photon fluence...")
cuda.memcpy_htod(fluence_grid_gpu, fluence_grid)
cuda.memcpy_htod(oad_grid_gpu, oad_grid)
cuda.memcpy_htod(blocked_grid_gpu, blocked_grid)
cuda.memcpy_htod(num_voxels_gpu, grid.num_voxels)
cuda.memcpy_htod(corner_gpu, grid.corner)
cuda.memcpy_htod(resolution_gpu, grid.resolution)
cuda.memcpy_htod(source_position_gpu, source.position)
cuda.memcpy_htod(beam_profile_correction_fs_interp_gpu, beam_profile_correction_fs_interp)
fluence(
    fluence_grid_gpu,
    oad_grid_gpu,
    blocked_grid_gpu,
    num_voxels_gpu,
    corner_gpu,
    resolution_gpu,
    source_position_gpu,
    beam_profile_correction_fs_interp_gpu,
    np.float32(beam_profile_correction_dx),
    np.float32(source.sad),
    np.float32(settings["sources"]["s_pri"]),
    np.float32(settings["sources"]['s_ann']),
    np.float32(settings["sources"]['z_ann']),
    np.float32(settings["sources"]['r_inner']),
    np.float32(settings["sources"]['r_outer']),
    np.float32(settings["sources"]['z_exp']),
    np.float32(settings["sources"]['s_exp']),
    np.float32(settings["sources"]['k_exp']),
    block=threadsperblock,
    grid=blockspergrid
)
cuda.memcpy_dtoh(fluence_grid, fluence_grid_gpu)

# %%
print("Calculating TERMA...")
cuda.memcpy_htod(terma_grid_gpu, terma_grid)
cuda.memcpy_htod(blocked_grid_gpu, blocked_grid)
cuda.memcpy_htod(fluence_grid_gpu, fluence_grid)
cuda.memcpy_htod(d_eff_grid_gpu, d_eff_grid)
cuda.memcpy_htod(num_voxels_gpu, grid.num_voxels)
cuda.memcpy_htod(energies_gpu, energies)
cuda.memcpy_htod(energy_weights_gpu, energy_weights)
cuda.memcpy_htod(mu_w_gpu, mu_w)
cuda.memcpy_htod(oad_grid_gpu, oad_grid)
cuda.memcpy_htod(off_axis_softening_fs_interp_gpu, off_axis_softening_fs_interp)
terma(
    terma_grid_gpu,
    blocked_grid_gpu,
    fluence_grid_gpu,
    d_eff_grid_gpu,
    num_voxels_gpu,
    np.int32(len(energies)),
    energies_gpu,
    energy_weights_gpu,
    mu_w_gpu,
    oad_grid_gpu,
    off_axis_softening_fs_interp_gpu,
    np.float32(off_axis_softening_dx),
    block=threadsperblock,
    grid=blockspergrid
)
cuda.memcpy_dtoh(terma_grid, terma_grid_gpu)

# tmp = terma_grid[50, :, 50]
# terma_grid = np.zeros_like(density_grid, dtype=np.float32)
# terma_grid[50, :, 50] = tmp  # Test TERMA
terma_grid = np.zeros_like(density_grid, dtype=np.float32)
terma_grid[50, 50, 50] = 1.0  # Test TERMA

# %%
if settings["calculation"]["mask_enable"]:
    print("Calculating mask...")
    cuda.memcpy_htod(mask_grid_gpu, mask_grid)
    cuda.memcpy_htod(num_voxels_gpu, grid.num_voxels)
    cuda.memcpy_htod(corner_gpu, grid.corner)
    cuda.memcpy_htod(resolution_gpu, grid.resolution)
    cuda.memcpy_htod(source_position_gpu, source.position)
    mask(
        mask_grid_gpu,
        terma_grid_gpu,
        num_voxels_gpu,
        corner_gpu,
        resolution_gpu,
        np.float32(settings["calculation"]["mask_max_distance"]),
        np.float32(terma_grid.max() * settings["calculation"]["mask_terma_threshold"]),
        block=threadsperblock,
        grid=blockspergrid
    )
    cuda.memcpy_dtoh(mask_grid, mask_grid_gpu)
else:
    print("Skipping mask calculation.")
    mask_grid = np.ones(density_grid.shape, dtype=np.float32)

# %%
print("Calculating dose...")
cuda.memcpy_htod(dose_grid_gpu, dose_grid)
cuda.memcpy_htod(resolution_gpu, grid.resolution)
cuda.memcpy_htod(num_voxels_gpu, grid.num_voxels)
cuda.memcpy_htod(corner_gpu, grid.corner)
cuda.memcpy_htod(density_grid_gpu, density_grid)
cuda.memcpy_htod(terma_grid_gpu, terma_grid)
cuda.memcpy_htod(mask_grid_gpu, mask_grid)
cuda.memcpy_htod(kernel_thetas_gpu, kernel_thetas)
cuda.memcpy_htod(kernel_phis_c_gpu, kernel_phis_c)
cuda.memcpy_htod(kernel_gpu, kernel)
cuda.memcpy_htod(source_v_x_gpu, source.v_x)
cuda.memcpy_htod(source_v_y_gpu, source.v_y)
cuda.memcpy_htod(source_v_z_gpu, source.v_z)
# start_evt = cuda.Event()
# end_evt = cuda.Event()
# start_evt.record()
dose(
    dose_grid_gpu,
    resolution_gpu,
    num_voxels_gpu,
    corner_gpu,
    density_grid_gpu,
    terma_grid_gpu,
    mask_grid_gpu,
    kernel_thetas_gpu,
    kernel_phis_c_gpu,
    kernel_gpu,
    source_v_x_gpu,
    source_v_y_gpu,
    source_v_z_gpu,
    np.int32(1192),     # n depth bins
    np.float32(0.05),   # float32 (e.g. 0.025)
    np.float32(59.6),      # float32 (eg n_depth_bins * depth_res)
    np.float32(0.05),    # float32 (ray march step, e.g. 0.025)
    block=threadsperblock,
    grid=blockspergrid
)
# end_evt.record()
# end_evt.synchronize()
# elapsed_time_ms = start_evt.time_till(end_evt)
# print(f"GPU operation took {elapsed_time_ms:.3f} ms")
cuda.memcpy_dtoh(dose_grid, dose_grid_gpu)

# # %%
# print("Calculating active dose...")
# cuda.memcpy_htod(dose_grid_gpu, dose_grid)
# cuda.memcpy_htod(resolution_gpu, grid.resolution)
# cuda.memcpy_htod(num_voxels_gpu, grid.num_voxels)
# cuda.memcpy_htod(corner_gpu, grid.corner)
# cuda.memcpy_htod(density_grid_gpu, density_grid)
# cuda.memcpy_htod(terma_grid_gpu, terma_grid)
# cuda.memcpy_htod(mask_grid_gpu, mask_grid)
# cuda.memcpy_htod(kernel_thetas_gpu, kernel_thetas)
# cuda.memcpy_htod(kernel_phis_c_gpu, kernel_phis_c)
# cuda.memcpy_htod(kernel_gpu, kernel)
# cuda.memcpy_htod(source_v_x_gpu, source.v_x)
# cuda.memcpy_htod(source_v_y_gpu, source.v_y)
# cuda.memcpy_htod(source_v_z_gpu, source.v_z)
# start_evt = cuda.Event()
# end_evt = cuda.Event()
# start_evt.record()
# active_dose(
#     dose_grid_gpu,
#     resolution_gpu,
#     num_voxels_gpu,
#     corner_gpu,
#     density_grid_gpu,
#     terma_grid_gpu,
#     mask_grid_gpu,
#     kernel_thetas_gpu,
#     kernel_phis_c_gpu,
#     kernel_gpu,
#     source_v_x_gpu,
#     source_v_y_gpu,
#     source_v_z_gpu,
#     np.int32(800),     # n depth bins
#     np.float32(0.025),   # float32 (e.g. 0.025)
#     np.float32(20),      # float32 (eg n_depth_bins * depth_res)
#     np.float32(0.025),    # float32 (ray march step, e.g. 0.025)
#     np.int32(settings["calculation"]["active_dose_interp_skip"]),
#     np.float32(settings["calculation"]["active_dose_interp_terma_threshold"] * terma_grid.max()),
#     np.float32(settings["calculation"]["active_dose_interp_dose_threshold"]),
#     block=threadsperblock,
#     grid=blockspergrid
# )
# end_evt.record()
# end_evt.synchronize()
# elapsed_time_ms = start_evt.time_till(end_evt)
# print(f"GPU operation took {elapsed_time_ms:.3f} ms")
# cuda.memcpy_dtoh(dose_grid, dose_grid_gpu)




# %%
import pandas as pd
file_path = '6MV Beam Data.xlsx'
df = pd.read_excel(file_path, sheet_name='Open Field Depth Dose')
xs = np.linspace(0, 40, 201) + 0.1
plt.plot(xs, dose_grid[100,:,100]/dose_grid[100,:,100].max() * 100)
plt.plot(df.iloc[5:, 0], df.iloc[5:, 5])
# %%
