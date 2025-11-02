# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator, make_interp_spline
import sys
import os
import toml
from importlib.resources import files
from conehead.kernel import KernelMono
from conehead.nist import mu_water
from conehead.source import Source
from conehead.phantom import SimplePhantom
from conehead.block import Block

sys.path.append(os.path.join(os.getcwd(), "../build/"))
import conehead_gpu as gpu
import time

settings = toml.load("Truebeam_6FFF_M120.toml")

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


# energies = np.array(settings["energy_spectrum"]["energies"], dtype=np.float32)
# energy_weights = np.array(settings["energy_spectrum"]["weights"], dtype=np.float32)
# mu = 1.6
# sigma = 1.6
# energy_weights = 1 / (np.sqrt(2 * np.pi) * sigma * energies)
# energy_weights *= np.exp(-((np.log(energies) - mu) ** 2) / (2 * sigma**2))

# # energy_weights = np.power(energies, C1 - 1) * np.exp(-C2 * energies)
# # energy_weights = energy_weights.astype(np.float32)
# N = np.sum(energy_weights)
# energy_weights /= N
# energy_weights /= energies


oads = np.array([0.0, 40.0], dtype=np.float32)
off_axis_softening_fs_interp = np.array([0.0, 0.0], dtype=np.float32)
mu_w = mu_water(energies)

phantom = SimplePhantom()
# phantom.densities[:, 25:71, :] = np.float32(0.2813)  # Feature

block = Block()
block.set_square(np.float32(10))
x_orig = np.linspace(-20.0, 20.0, 4000)  # 4000 points from -20 to 20
y_orig = np.linspace(-20.0, 20.0, 4000)  # 4000 points from -20 to 20
X_orig, Y_orig = np.meshgrid(x_orig, y_orig)
x_target = np.linspace(-28.0, 28.0, 560)  # 560 points from -28 to 28
y_target = np.linspace(-28.0, 28.0, 560)  # 560 points from -28 to 28
X_target, Y_target = np.meshgrid(x_target, y_target)
interpolator = RegularGridInterpolator(
    (x_orig, y_orig), block.block_values, method="linear", bounds_error=False, fill_value=0
)
points_target = np.array([X_target.ravel(), Y_target.ravel()]).T
block_interpolated = interpolator(points_target)
block_interpolated_2d = block_interpolated.reshape(X_target.shape)
pixel_pitch_cm = 0.1  # cm
sigma_pix_x = settings["sources"]["pri_x"] / pixel_pitch_cm
sigma_pix_y = settings["sources"]["pri_y"] / pixel_pitch_cm
pri_fluence = gaussian_filter(
    block_interpolated_2d, sigma=(sigma_pix_x, sigma_pix_y), mode="nearest"
)
sigma_pix_x = settings["sources"]["sec_x"] / pixel_pitch_cm
sigma_pix_y = settings["sources"]["sec_y"] / pixel_pitch_cm
sec_fluence = gaussian_filter(
    block_interpolated_2d, sigma=(sigma_pix_x, sigma_pix_y), mode="nearest"
)
bpc_interp = make_interp_spline(
    settings["beam_profile_correction"]["oads"], settings["beam_profile_correction"]["factors"], k=1
)
x = np.arange(-28, 28, 0.1, dtype=np.float32)
y = np.arange(-28, 28, 0.1, dtype=np.float32)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
bpc = bpc_interp(r)
fluence_map_pri = settings["sources"]["pri_s"] * pri_fluence * bpc
fluence_map_sec = settings["sources"]["sec_s"] * sec_fluence
fluence_map_pri = fluence_map_pri.astype(np.float32)
fluence_map_sec = fluence_map_sec.astype(np.float32)

oad_grid = np.zeros(phantom.num_voxels, dtype=np.float32)
d_geo_grid = np.zeros(phantom.num_voxels, dtype=np.float32)
d_eff_grid = np.zeros(phantom.num_voxels, dtype=np.float32)
density_grid = phantom.densities
fluence_grid = np.zeros(phantom.num_voxels, dtype=np.float32)
# fluence_map_pri = np.zeros((560, 560), dtype=np.float32) * pri_s
# fluence_map_pri[250:310, 250:310] = 1.0
# fluence_map_pri = gaussian_filter(fluence_map_pri, sigma=(2, 2), mode="nearest")
# fluence_map_pri = fluence_map_pri.astype(np.float32)
# fluence_map_sec = np.zeros((560, 560), dtype=np.float32) * sec_s
# fluence_map_sec[250:310, 250:310] = 1.0
# fluence_map_sec = gaussian_filter(fluence_map_sec, sigma=(50, 50), mode="nearest")
# fluence_map_sec = fluence_map_sec.astype(np.float32)
terma_grid = np.zeros(phantom.num_voxels, dtype=np.float32)
mask_grid = np.zeros(phantom.num_voxels, dtype=np.float32)
dose_grid = np.zeros(phantom.num_voxels, dtype=np.float32)

source = Source()
# source.gantry = np.float32(45.0)

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


# d_geo_grid_py = np.zeros(phantom.num_voxels, dtype=np.float32)
# for i in range(phantom.num_voxels[0]):
#     for j in range(phantom.num_voxels[1]):
#         for k in range(phantom.num_voxels[2]):
#             pos_x = phantom.corner[0] + i * phantom.resolution[0] + phantom.resolution[0] / 2
#             pos_y = phantom.corner[1] + j * phantom.resolution[1] + phantom.resolution[1] / 2
#             pos_z = phantom.corner[2] + k * phantom.resolution[2] + phantom.resolution[2] / 2
#             voxel_pos = np.array([pos_x, pos_y, pos_z], dtype=np.float32)
#             vec_source_to_voxel = voxel_pos - source.position
#             distance = np.sqrt(
#                 vec_source_to_voxel[0] ** 2
#                 + vec_source_to_voxel[1] ** 2
#                 + vec_source_to_voxel[2] ** 2
#             )
#             d_geo_grid_py[i, j, k] = distance


runs = 1
t0 = time.time()
for _ in range(runs):
    gpu.oad(
        oad_grid=oad_grid,
        num_voxels=phantom.num_voxels,
        corner=phantom.corner,
        resolution=phantom.resolution,
        source_position=source.position,
        source_v_x=source.v_x,
        source_v_y=source.v_y,
        source_v_z=source.v_z,
    )
print("oad time: " + str((time.time() - t0) / runs) + " s")


runs = 1
t0 = time.time()
for _ in range(runs):
    gpu.d_geo(
        d_geo_grid=d_geo_grid,
        num_voxels=phantom.num_voxels,
        corner=phantom.corner,
        resolution=phantom.resolution,
        source_position=source.position,
    )
print("d_geo time: " + str((time.time() - t0) / runs) + " s")


runs = 1
t0 = time.time()
for _ in range(runs):
    gpu.d_eff(
        d_eff_grid=d_eff_grid,
        num_voxels=phantom.num_voxels,
        corner=phantom.corner,
        resolution=phantom.resolution,
        density_grid=density_grid,
        source_position=source.position,
    )
print("d_eff time: " + str((time.time() - t0) / runs) + " s")


runs = 1
t0 = time.time()
for _ in range(runs):
    gpu.fluence(
        fluence_grid=fluence_grid,
        fluence_map_pri=fluence_map_pri,
        fluence_map_sec=fluence_map_sec,
        num_voxels=phantom.num_voxels,
        corner=phantom.corner,
        resolution=phantom.resolution,
        d_geo_grid=d_geo_grid,
        source_position=source.position,
        source_v_x=source.v_x,
        source_v_y=source.v_y,
        source_v_z=source.v_z,
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


runs = 1
t0 = time.time()
for _ in range(runs):
    gpu.terma(
        terma_grid=terma_grid,
        fluence_grid=fluence_grid,
        d_geo_grid=d_geo_grid,
        d_eff_grid=d_eff_grid,
        num_voxels=phantom.num_voxels,
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

# terma_grid = np.zeros_like(terma_grid, dtype=np.float32)  # reset terma grid for testing
# terma_grid[100, 100, 100] = 1.0  # point terma for testing dose


runs = 1
t0 = time.time()
for _ in range(runs):
    gpu.mask(
        mask_grid=mask_grid,
        terma_grid=terma_grid,
        num_voxels=phantom.num_voxels,
        resolution=phantom.resolution,
        max_distance_cm=mask_max_distance,
        terma_threshold=mask_terma_threshold * terma_grid.max(),
    )
print("mask time: " + str((time.time() - t0) / runs) + " s")


mask_grid = np.ones_like(mask_grid, dtype=np.float32)  # for testing dose without mask


runs = 1
t0 = time.time()
for _ in range(runs):
    gpu.dose(
        dose_grid=dose_grid,
        resolution=phantom.resolution,
        num_voxels=phantom.num_voxels,
        corner=phantom.corner,
        density_grid=density_grid,
        d_geo_grid=d_geo_grid,
        terma_grid=terma_grid,
        mask_grid=mask_grid,
        kernel_thetas=kernel_thetas,
        kernel_phis=kernel_phis_c,
        kernel_omegas=kernel_omegas,
        kernel=kernel,
        source_sad=source_sad,
        source_position=source.position,
        source_v_x=source.v_x,
        source_v_y=source.v_y,
        source_v_z=source.v_z,
        n_depth_bins=np.int32(1192),
        kernel_depth_res_cm=np.float32(0.05),
        max_kernel_depth_cm=np.float32(59.6),
        ds_cm=np.float32(0.05),
    )
print("dose time: " + str((time.time() - t0) / 3) + " s")


x_gap = np.abs(block.x2_jaw_pos - block.x1_jaw_pos)
y_gap = np.abs(block.y2_jaw_pos - block.y1_jaw_pos)

field_measure = 2 * x_gap * y_gap / (x_gap + y_gap)
ofc = np.interp(
    field_measure,
    settings["output_factor_correction"]["field_sizes"],
    settings["output_factor_correction"]["factors"],
).astype(np.float32)
N = np.float32(settings["calculation"]["normalisation"])
MU = np.float32(100.0)
dose_grid = dose_grid * ofc * N * MU

hx = phantom.num_voxels[0] // 2
hy = phantom.num_voxels[1] // 2
hz = phantom.num_voxels[2] // 2

fig, ax = plt.subplots(3, 2, figsize=(12, 16))
ax[0, 0].set_title("oad")
ax[0, 0].imshow(oad_grid[hz, :, :], interpolation="nearest")
ax[0, 1].set_title("d_geo")
ax[0, 1].imshow(d_geo_grid[hz, :, :], interpolation="nearest")
ax[1, 0].set_title("d_eff")
ax[1, 0].imshow(d_eff_grid[hz, :, :], interpolation="nearest")
ax[1, 1].set_title("fluence")
ax[1, 1].imshow(fluence_grid[hz, :, :], interpolation="nearest")
ax[2, 0].set_title("terma")
ax[2, 0].imshow(terma_grid[hz, :, :], interpolation="nearest")
ax[2, 1].set_title("dose")
ax[2, 1].imshow(dose_grid[hz, :, :], interpolation="nearest")


# %%
class Curve:
    def __init__(self, x, y, type, size, of, depth=0.0):
        self.x = x
        self.y = y
        self.type = type
        self.size = size
        self.depth = depth
        self.of = of


import pandas as pd

file_path = "6FFF Beam Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Open Field Depth Dose")
measured_03x03_pdd = Curve(
    x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 1]), type="depth", size=3, of=0.8432
)
measured_04x04_pdd = Curve(
    x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 2]), type="depth", size=4, of=0.8753
)
measured_06x06_pdd = Curve(
    x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 3]), type="depth", size=6, of=0.9285
)
measured_08x08_pdd = Curve(
    x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 4]), type="depth", size=8, of=0.9704
)
measured_10x10_pdd = Curve(
    x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 5]), type="depth", size=10, of=1.000
)
measured_20x20_pdd = Curve(
    x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 6]), type="depth", size=20, of=1.0837
)
measured_30x30_pdd = Curve(
    x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 7]), type="depth", size=30, of=1.1190
)
measured_40x40_pdd = Curve(
    x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 8]), type="depth", size=40, of=1.1349
)
xs = np.linspace(0, 40, 201) + 0.1
plt.plot(
    measured_10x10_pdd.x, measured_10x10_pdd.y / measured_10x10_pdd.y.max(), label="Measured 10x10"
)
plt.plot(xs, dose_grid[hx, :, hz] / dose_grid[hx, :, hz].max(), label="Calculated 10x10")
# oad time: 0.00802924633026123 s
# d_geo time: 0.007828712463378906 s
# d_eff time: 0.03479955196380615 s
# fluence time: 0.02115046977996826 s
# terma time: 0.05031294822692871 s
# mask time: 0.011710929870605468 s
# dose time: 5.819006522496541 s

# %%
