# %%
import numpy as np
import matplotlib.pyplot as plt
import toml
import math
import pycuda.driver as cuda
import pycuda.autoinit
import shutil
from scipy.optimize import minimize
import pandas as pd
from pycuda.compiler import SourceModule
from conehead.kernel import KernelMono
from conehead.phantom import SimplePhantom
from conehead.source import Source
from conehead.dosegrid import DoseGrid
from conehead.block import Block
from conehead.nist import mu_water

file_path = '6MV Beam Data.xlsx'
df = pd.read_excel(file_path, sheet_name='Open Field Depth Dose')

# def optimise_me(x):

## %%
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

# x = [5.709e-01,  1.004e-01]

# # FOR OPTIMISATION ONLY
# w = 1 / (np.sqrt(2 * np.pi) * x[1] * np.array(settings["energy_spectrum"]["energies"])) * np.exp(-(np.log(np.array(settings["energy_spectrum"]["energies"])) - x[0])**2 / (2 * x[1]**2))
# # c1 = x[0]/(x[0] - x[1])
# # c2 = 1 / (x[0] - x[1])
# # w = (np.array(settings["energy_spectrum"]["energies"]) ** (c1 - 1)) * np.exp(-c2 * np.array(settings["energy_spectrum"]["energies"]))
# settings["energy_spectrum"]["weights"] = w


# Collapse per-energy kernels into a single kernel using global spectrum weights
kernel = np.zeros_like(kernels[0].kernel, dtype=np.float32)
for i in range(len(settings["energy_spectrum"]["energies"])):
    kernel += kernels[i].kernel * settings["energy_spectrum"]["weights"][i]
kernel = kernel / kernel.sum()  # normalise
kernel_radii = kernels[0].radii_centres
kernel_phis = kernels[0].angles
kernel_phis_c = kernels[0].angles_centres
kernel_thetas = np.linspace(0, 360 - (360 / 16), 16, dtype=np.float32)  # 16 thetas
kernel = kernel / len(kernel_thetas)  # account for theta sampling
kernel_omegas = (kernels[0].omegas / len(kernel_thetas)).astype(np.float32)

## %%
# Build kernel bank across water-equivalent depth T in [-50, +50] cm, step 1 cm.
# This does not alter the current pipeline; it prepares `kernel_bank` for later use.
# Convention: each banked kernel is normalized to sum to 1 over all (phi, depth) bins,
# consistent with the current single collapsed kernel usage.
# Water-equivalent depth grid (cm)
kernel_bank_T_cm = np.arange(-50.0, 51.0, 1.0, dtype=np.float32)  # [-50, -49, ..., 50]
# Stack monoenergetic kernels: shape (nE, nPhi, nDepth)
kernels_stack = np.stack([k.kernel for k in kernels]).astype(np.float32)
nE, nPhi, nDepth = kernels_stack.shape
# Base normalized spectrum weights (shape nE) from settings
energies_arr = np.array(settings["energy_spectrum"]["energies"], dtype=np.float32)
w0 = np.array(settings["energy_spectrum"]["weights"], dtype=np.float32).astype(np.float32)
w0 = w0 / w0.sum()
# Compute water attenuation coefficients for these energies
mu_w_arr = mu_water(energies_arr).astype(np.float32)
# Compute weights for each T: W[T, e] = w0[e] * exp(-mu_w[e] * T), then normalize across energies
W = w0[None, :] * np.exp(-mu_w_arr[None, :] * kernel_bank_T_cm[:, None]).astype(np.float32)
W = W / W.sum(axis=1, keepdims=True)
# Mix kernels across energies for each T: result shape (nT, nPhi, nDepth)
kernel_bank = (W @ kernels_stack.reshape(nE, -1)).reshape(len(kernel_bank_T_cm), nPhi, nDepth)
# Normalize each kernel in the bank to sum to 1 (keeps behavior consistent with current code)
kernel_bank = kernel_bank / kernel_bank.reshape(len(kernel_bank_T_cm), -1).sum(axis=1)[:, None, None]
# Optional: quick sanity print (commented to keep output clean)
# print(f"Kernel bank built: T bins = {len(kernel_bank_T_cm)}, shape per kernel = ({nPhi}, {nDepth})")

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
block.set_square(np.float32(20)) 

energies = np.array([np.float32(x) for x in settings["energy_spectrum"]["energies"]], dtype=np.float32)
energy_weights = np.array([np.float32(x) for x in settings["energy_spectrum"]["weights"]], dtype=np.float32)
energy_weights = energy_weights / energy_weights.sum()  # normalise
mu_w = mu_water(energies)


## %%
# Create python data arrays
density_grid = phantom.densities
blocked_grid = np.ones(density_grid.shape, dtype=np.float32)
oad_grid = np.zeros(density_grid.shape, dtype=np.float32)
d_geo_grid = np.zeros(density_grid.shape, dtype=np.float32)
d_eff_grid = np.zeros(density_grid.shape, dtype=np.float32)
fluence_grid = np.zeros(density_grid.shape, dtype=np.float32)
terma_grid = np.zeros(density_grid.shape, dtype=np.float32)
mask_grid = np.zeros(density_grid.shape, dtype=np.float32)
dose_grid = np.zeros(density_grid.shape, dtype=np.float32)

# Allocate GPU memory
density_grid_gpu = cuda.mem_alloc(density_grid.nbytes)
blocked_grid_gpu = cuda.mem_alloc(density_grid.nbytes)
oad_grid_gpu = cuda.mem_alloc(density_grid.nbytes)
d_geo_grid_gpu = cuda.mem_alloc(density_grid.nbytes)
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
kernel_omegas_gpu = cuda.mem_alloc(kernel_omegas.nbytes)
kernel_gpu = cuda.mem_alloc(kernel.nbytes)
kernel_bank_gpu = cuda.mem_alloc(kernel_bank.nbytes)


# Compile the CUDA kernel
cuda_code = open("conehead.cu").read()
mod = SourceModule(cuda_code)

# Extract kernel functions
hit_test = mod.get_function("hit_test")
oad = mod.get_function("oad")
d_geo = mod.get_function("d_geo")
d_eff = mod.get_function("d_eff")
fluence = mod.get_function("fluence")
terma = mod.get_function("terma")
mask = mod.get_function("mask")
dose = mod.get_function("dose")
dose_banked = mod.get_function("dose_banked")
# active_dose = mod.get_function("active_dose")

# Define grid/block sizes
threadsperblock = (16, 4, 4)
blockspergrid_x = math.ceil(density_grid.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(density_grid.shape[1] / threadsperblock[1])
blockspergrid_z = math.ceil(density_grid.shape[2] / threadsperblock[2])
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

## %%
# print("Performing hit-testing of dose grid voxels...")
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

## %%
# print("Calculating off-axis distances")
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

## %%
# print("Calculating geometric depths...")
cuda.memcpy_htod(d_geo_grid_gpu, d_geo_grid)
cuda.memcpy_htod(num_voxels_gpu, grid.num_voxels)
cuda.memcpy_htod(corner_gpu, grid.corner)
cuda.memcpy_htod(resolution_gpu, grid.resolution)
cuda.memcpy_htod(density_grid_gpu, density_grid)
cuda.memcpy_htod(source_position_gpu, source.position)
d_geo(
    d_geo_grid_gpu,
    num_voxels_gpu,
    corner_gpu,
    resolution_gpu,
    density_grid_gpu,
    source_position_gpu,
    block=threadsperblock,
    grid=blockspergrid,
)
cuda.memcpy_dtoh(d_geo_grid, d_geo_grid_gpu)

## %%
# print("Calculating effective depths...")
cuda.memcpy_htod(d_eff_grid_gpu, d_eff_grid)
cuda.memcpy_htod(num_voxels_gpu, grid.num_voxels)
cuda.memcpy_htod(corner_gpu, grid.corner)
cuda.memcpy_htod(resolution_gpu, grid.resolution)
cuda.memcpy_htod(density_grid_gpu, density_grid)
cuda.memcpy_htod(source_position_gpu, source.position)
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
cuda.memcpy_dtoh(d_eff_grid, d_eff_grid_gpu)

## %%
# print("Calculating photon fluence...")
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

## %%
# print("Calculating TERMA...")
cuda.memcpy_htod(terma_grid_gpu, terma_grid)
cuda.memcpy_htod(blocked_grid_gpu, blocked_grid)
cuda.memcpy_htod(fluence_grid_gpu, fluence_grid)
cuda.memcpy_htod(d_geo_grid_gpu, d_geo_grid)
cuda.memcpy_htod(d_eff_grid_gpu, d_eff_grid)
cuda.memcpy_htod(num_voxels_gpu, grid.num_voxels)
cuda.memcpy_htod(energies_gpu, energies)
cuda.memcpy_htod(energy_weights_gpu, energy_weights)
cuda.memcpy_htod(mu_w_gpu, mu_w)
cuda.memcpy_htod(source_position_gpu, source.position)
cuda.memcpy_htod(oad_grid_gpu, oad_grid)
cuda.memcpy_htod(off_axis_softening_fs_interp_gpu, off_axis_softening_fs_interp)
terma(
    terma_grid_gpu,
    blocked_grid_gpu,
    fluence_grid_gpu,
    d_geo_grid_gpu,
    d_eff_grid_gpu,
    num_voxels_gpu,
    np.int32(len(energies)),
    energies_gpu,
    energy_weights_gpu,
    mu_w_gpu,
    np.float32(source.sad),
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

# terma_grid = np.zeros_like(density_grid, dtype=np.float32)
# terma_grid[50, 50, 50] = 1.0  # Test TERMA

## %%
if settings["calculation"]["mask_enable"]:
    # print("Calculating mask...")
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
    # print("Skipping mask calculation.")
    mask_grid = np.ones(density_grid.shape, dtype=np.float32)

# ## %%
# # print("Calculating dose...")
# cuda.memcpy_htod(dose_grid_gpu, dose_grid)
# cuda.memcpy_htod(resolution_gpu, grid.resolution)
# cuda.memcpy_htod(num_voxels_gpu, grid.num_voxels)
# cuda.memcpy_htod(corner_gpu, grid.corner)
# cuda.memcpy_htod(density_grid_gpu, density_grid)
# cuda.memcpy_htod(d_geo_grid_gpu, d_geo_grid)
# cuda.memcpy_htod(terma_grid_gpu, terma_grid)
# cuda.memcpy_htod(mask_grid_gpu, mask_grid)
# cuda.memcpy_htod(kernel_thetas_gpu, kernel_thetas)
# cuda.memcpy_htod(kernel_phis_c_gpu, kernel_phis_c)
# cuda.memcpy_htod(kernel_omegas_gpu, kernel_omegas)
# cuda.memcpy_htod(kernel_gpu, kernel)
# cuda.memcpy_htod(source_position_gpu, source.position)
# cuda.memcpy_htod(source_v_x_gpu, source.v_x)
# cuda.memcpy_htod(source_v_y_gpu, source.v_y)
# cuda.memcpy_htod(source_v_z_gpu, source.v_z)
# dose(
#     dose_grid_gpu,
#     resolution_gpu,
#     num_voxels_gpu,
#     corner_gpu,
#     density_grid_gpu,
#     d_geo_grid_gpu,
#     terma_grid_gpu,
#     mask_grid_gpu,
#     kernel_thetas_gpu,
#     kernel_phis_c_gpu,
#     kernel_omegas_gpu,
#     kernel_gpu,
#     np.float32(source.sad),
#     source_position_gpu,
#     source_v_x_gpu,
#     source_v_y_gpu,
#     source_v_z_gpu,
#     np.int32(1192),     # n depth bins
#     np.float32(0.05),   # kernel depth resolution (cm)
#     np.float32(59.6),   # max kernel depth (cm)
#     np.float32(0.05),   # ray-march step (cm)
#     block=threadsperblock,
#     grid=blockspergrid
# )
# cuda.memcpy_dtoh(dose_grid, dose_grid_gpu)



# print("Calculating dose (kernel bank)...")
cuda.memcpy_htod(dose_grid_gpu, dose_grid)
cuda.memcpy_htod(resolution_gpu, grid.resolution)
cuda.memcpy_htod(num_voxels_gpu, grid.num_voxels)
cuda.memcpy_htod(corner_gpu, grid.corner)
cuda.memcpy_htod(density_grid_gpu, density_grid)
cuda.memcpy_htod(d_geo_grid_gpu, d_geo_grid)
cuda.memcpy_htod(d_eff_grid_gpu, d_eff_grid)
cuda.memcpy_htod(terma_grid_gpu, terma_grid)
cuda.memcpy_htod(mask_grid_gpu, mask_grid)
cuda.memcpy_htod(oad_grid_gpu, oad_grid)
cuda.memcpy_htod(kernel_thetas_gpu, kernel_thetas)
cuda.memcpy_htod(kernel_phis_c_gpu, kernel_phis_c)
cuda.memcpy_htod(kernel_omegas_gpu, kernel_omegas)
cuda.memcpy_htod(kernel_bank_gpu, kernel_bank)
cuda.memcpy_htod(source_position_gpu, source.position)
cuda.memcpy_htod(source_v_x_gpu, source.v_x)
cuda.memcpy_htod(source_v_y_gpu, source.v_y)
cuda.memcpy_htod(source_v_z_gpu, source.v_z)
# Prepare bank/LUT parameters
n_T_bins = np.int32(kernel_bank.shape[0])
T_min = np.float32(kernel_bank_T_cm[0])
T_step = np.float32(kernel_bank_T_cm[1] - kernel_bank_T_cm[0]) if kernel_bank_T_cm.shape[0] > 1 else np.float32(1.0)
off_axis_table_len = np.int32(off_axis_softening_fs_interp.shape[0])
dose_banked(
    dose_grid_gpu,
    resolution_gpu,
    num_voxels_gpu,
    corner_gpu,
    density_grid_gpu,
    d_geo_grid_gpu,
    d_eff_grid_gpu,
    terma_grid_gpu,
    mask_grid_gpu,
    oad_grid_gpu,
    kernel_thetas_gpu,
    kernel_phis_c_gpu,
    kernel_omegas_gpu,
    # kernel bank
    kernel_bank_gpu,
    n_T_bins,
    T_min,
    T_step,
    # Off-axis softening LUT
    off_axis_softening_fs_interp_gpu,
    off_axis_table_len,
    np.float32(off_axis_softening_dx),
    # Geom/scales
    np.float32(source.sad),
    source_position_gpu,
    source_v_x_gpu,
    source_v_y_gpu,
    source_v_z_gpu,
    # Kernel sampling
    np.int32(1192),     # n depth bins
    np.float32(0.05),   # kernel depth resolution (cm)
    np.float32(59.6),   # max kernel depth (cm)
    np.float32(0.05),   # ray-march step (cm)
    block=threadsperblock,
    grid=blockspergrid
)
cuda.memcpy_dtoh(dose_grid, dose_grid_gpu)









#     ## %%
#     xs = np.linspace(0, 40, 201) + 0.1

#     gold = np.interp(xs, df.iloc[5:, 0].to_numpy().astype(np.float32), (df.iloc[5:, 5]/df.iloc[5:, 5].max() * 100).to_numpy().astype(np.float32))
#     # gold = np.interp(xs, df.iloc[5:, 0].to_numpy().astype(np.float32), (df.iloc[5:, 5]/df.iloc[105, 5] * 100).to_numpy().astype(np.float32))
#     calc = dose_grid[100, :, 100] / dose_grid[100, :, 100].max() * 100
#     # calc = dose_grid[100, :, 100] / dose_grid[100, 50, 100] * 100
#     diff = np.sum(np.abs(calc[10:149] - gold[10:149]))
#     print(f"Weights: {x}, Diff: {diff}")
#     return diff


# # %%
# x0 = [1.0, 1.0]
# bounds = [(0.1, 3) for _ in x0]
# constraints = ({
#     'type': 'ineq', 'fun': lambda x: x[1] - x[0] - 0.01,  # x[1] must be greater than x[0]
# })


# # maxiter = 500
# # result = minimize(optimise_me, x0, bounds=bounds, constraints=constraints, method ='COBYLA', options={'disp': True})
# result = minimize(optimise_me, x0, bounds=bounds, method ='COBYLA', options={'disp': True})

# # Print the optimization result
# print(f"Optimization Result: {result}")

# # # file_path = '6MV Beam Data.xlsx'
# # # df = pd.read_excel(file_path, sheet_name='Open Field Depth Dose')
# # # xs = np.linspace(0, 40, 201) + 0.1
# # # plt.plot(xs, dose_grid[100,:,100]/dose_grid[100,:,100].max() * 100)
# # # plt.plot(df.iloc[5:, 0], df.iloc[5:, 5])
















# %%
