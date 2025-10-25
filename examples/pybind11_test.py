# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import sys
import os
import toml
from importlib.resources import files
from conehead.kernel import KernelMono
from conehead.nist import mu_water

sys.path.append(os.path.join(os.getcwd(), "../build/"))
import conehead_gpu as gpu
import time

settings = toml.load("settings_6FFF.toml")

source_sad = np.float32(100.0)
pri_s = np.float32(0.98373769)
pri_x = np.float32(0.14093192)
pri_y = np.float32(0.14093192)
pri_z = np.float32(0.5)
sec_s = np.float32(0.01626231)
sec_x = np.float32(25.26913868)
sec_y = np.float32(25.26913868)
sec_z = np.float32(20.0)
samples = np.int32(3)
mask_max_distance = np.float32(5.0)  # cm
mask_terma_threshold = np.float32(0.005)

energies = settings["energy_spectrum"]["energies"]
energy_weights = settings["energy_spectrum"]["weights"]

oads = np.array([0.0, 40.0], dtype=np.float32)
off_axis_softening_fs_interp = np.array([0.0, 0.0], dtype=np.float32)
mu_w = mu_water(energies)

num_voxels = np.array([301, 301, 301], dtype=np.int32)
corner = np.array([-30.1, 0, -30.1], dtype=np.float32)
resolution = np.array([0.2, 0.2, 0.2], dtype=np.float32)
source_position = np.array([0.0, -100.0, 0.0], dtype=np.float32)
source_v_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
source_v_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
source_v_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

oad_grid = np.zeros(num_voxels, dtype=np.float32)
d_geo_grid = np.zeros(num_voxels, dtype=np.float32)
d_eff_grid = np.zeros(num_voxels, dtype=np.float32)
density_grid = np.ones(num_voxels, dtype=np.float32) * 1.0  # g/cm3
fluence_grid = np.zeros(num_voxels, dtype=np.float32)
fluence_map_pri = np.zeros((560, 560), dtype=np.float32) * pri_s
fluence_map_pri[100:460, 100:460] = 1.0
fluence_map_pri = gaussian_filter(fluence_map_pri, sigma=(2, 2), mode="nearest")
fluence_map_sec = np.zeros((560, 560), dtype=np.float32) * sec_s
fluence_map_sec[100:460, 100:460] = 1.0
fluence_map_sec = gaussian_filter(fluence_map_sec, sigma=(50, 50), mode="nearest")
terma_grid = np.zeros(num_voxels, dtype=np.float32)
mask_grid = np.zeros(num_voxels, dtype=np.float32)
dose_grid = np.zeros(num_voxels, dtype=np.float32)

kernels = [
    KernelMono(files("conehead.kernels").joinpath("0.5MeV/0.5MeV.egslst")),
    KernelMono(files("conehead.kernels").joinpath("1.0MeV/1.0MeV.egslst")),
    KernelMono(files("conehead.kernels").joinpath("1.5MeV/1.5MeV.egslst")),
    KernelMono(files("conehead.kernels").joinpath("2.0MeV/2.0MeV.egslst")),
    KernelMono(files("conehead.kernels").joinpath("2.5MeV/2.5MeV.egslst")),
    KernelMono(files("conehead.kernels").joinpath("3.0MeV/3.0MeV.egslst")),
    KernelMono(files("conehead.kernels").joinpath("3.5MeV/3.5MeV.egslst")),
    KernelMono(files("conehead.kernels").joinpath("4.0MeV/4.0MeV.egslst")),
    KernelMono(files("conehead.kernels").joinpath("4.5MeV/4.5MeV.egslst")),
    KernelMono(files("conehead.kernels").joinpath("5.0MeV/5.0MeV.egslst")),
    KernelMono(files("conehead.kernels").joinpath("5.5MeV/5.5MeV.egslst")),
    KernelMono(files("conehead.kernels").joinpath("6.0MeV/6.0MeV.egslst")),
]
kernel = np.zeros_like(kernels[0].kernel, dtype=np.float32)
for i in range(len(settings["energy_spectrum"]["energies"])):
    kernel += (
        kernels[i].kernel
        * settings["energy_spectrum"]["weights"][i]
        * settings["energy_spectrum"]["energies"][i]
    )
kernel = kernel / kernel.sum()  # normalise
kernel_radii = kernels[0].radii_centres
kernel_phis = kernels[0].angles
kernel_phis_c = kernels[0].angles_centres
kernel_thetas = np.linspace(0, 360 - (360 / 16), 16, dtype=np.float32)  # 16 thetas
kernel = kernel / len(kernel_thetas)  # account for theta sampling
kernel_omegas = (kernels[0].omegas / len(kernel_thetas)).astype(np.float32)


runs = 10
t0 = time.time()
for _ in range(runs):
    gpu.oad(
        oad_grid=oad_grid,
        num_voxels=num_voxels,
        corner=corner,
        resolution=resolution,
        source_position=source_position,
        source_v_x=source_v_x,
        source_v_y=source_v_y,
        source_v_z=source_v_z,
    )
print("oad time: " + str((time.time() - t0) / runs) + " s")


runs = 10
t0 = time.time()
for _ in range(runs):
    gpu.d_geo(
        d_geo_grid=d_geo_grid,
        num_voxels=num_voxels,
        corner=corner,
        resolution=resolution,
        source_position=source_position,
    )
print("d_geo time: " + str((time.time() - t0) / runs) + " s")


runs = 10
t0 = time.time()
for _ in range(runs):
    gpu.d_eff(
        d_eff_grid=d_eff_grid,
        num_voxels=num_voxels,
        corner=corner,
        resolution=resolution,
        density_grid=density_grid,
        source_position=source_position,
    )
print("d_eff time: " + str((time.time() - t0) / runs) + " s")


runs = 10
t0 = time.time()
for _ in range(runs):
    gpu.fluence(
        fluence_grid=fluence_grid,
        fluence_map_pri=fluence_map_pri,
        fluence_map_sec=fluence_map_sec,
        num_voxels=num_voxels,
        corner=corner,
        resolution=resolution,
        d_geo_grid=d_geo_grid,
        source_position=source_position,
        source_v_x=source_v_x,
        source_v_y=source_v_y,
        source_v_z=source_v_z,
        source_sad=source_sad,
        pri_s=pri_s,
        pri_x=pri_x,
        pri_y=pri_y,
        pri_z=pri_z,
        sec_s=sec_s,
        sec_x=sec_x,
        sec_y=sec_y,
        sec_z=sec_z,
        samples=samples,
    )
print("fluence time: " + str((time.time() - t0) / runs) + " s")


runs = 10
t0 = time.time()
for _ in range(runs):
    gpu.terma(
        terma_grid=terma_grid,
        fluence_grid=fluence_grid,
        d_geo_grid=d_geo_grid,
        d_eff_grid=d_eff_grid,
        num_voxels=num_voxels,
        num_energies=np.int32(len(energies)),
        energies=energies,
        energy_weights=energy_weights,
        mu_w=mu_w,
        source_sad=source_sad,
        oad_grid=oad_grid,
        off_axis_softening_fs_interp=off_axis_softening_fs_interp,
        off_axis_softening_dx=np.float32(40.0),
    )
print("terma time: " + str((time.time() - t0) / runs) + " s")


runs = 10
t0 = time.time()
for _ in range(runs):
    gpu.mask(
        mask_grid=mask_grid,
        terma_grid=terma_grid,
        num_voxels=num_voxels,
        resolution=resolution,
        max_distance_cm=mask_max_distance,
        terma_threshold=mask_terma_threshold * terma_grid.max(),
    )
print("mask time: " + str((time.time() - t0) / runs) + " s")


# mask_grid = np.ones_like(mask_grid, dtype=np.float32)  # for testing dose without mask


runs = 3
t0 = time.time()
for _ in range(runs):
    gpu.dose(
        dose_grid=dose_grid,
        resolution=resolution,
        num_voxels=num_voxels,
        corner=corner,
        density_grid=density_grid,
        d_geo_grid=d_geo_grid,
        terma_grid=terma_grid,
        mask_grid=mask_grid,
        kernel_thetas=kernel_thetas,
        kernel_phis=kernel_phis_c,
        kernel_omegas=kernel_omegas,
        kernel=kernel,
        source_sad=source_sad,
        source_position=source_position,
        source_v_x=source_v_x,
        source_v_y=source_v_y,
        source_v_z=source_v_z,
        n_depth_bins=np.int32(1192),
        kernel_depth_res_cm=np.float32(0.05),
        max_kernel_depth_cm=np.float32(59.6),
        ds_cm=np.float32(0.05),
    )
print("dose time: " + str((time.time() - t0) / 3) + " s")


# %%

# oad time: 0.00802924633026123 s
# d_geo time: 0.007828712463378906 s
# d_eff time: 0.03479955196380615 s
# fluence time: 0.02115046977996826 s
# terma time: 0.05031294822692871 s
# mask time: 0.011710929870605468 s
# dose time: 5.819006522496541 s
