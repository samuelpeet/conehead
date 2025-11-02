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
import pandas as pd
from scipy.optimize import minimize

sys.path.append(os.path.join(os.getcwd(), "../build/"))
import conehead_gpu as gpu


def run(fs, x):
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

    # energies = np.array(settings["energy_spectrum"]["energies"], dtype=np.float32)
    # energy_weights = settings["energy_spectrum"]["weights"]

    energies = np.array(settings["energy_spectrum"]["energies"], dtype=np.float32)
    energy_weights = np.array(settings["energy_spectrum"]["weights"], dtype=np.float32)
    # # mu = 0.91712852
    # # sigma = 0.88657253
    # mu = x[0]
    # sigma = x[1]
    # energy_weights = 1 / (np.sqrt(2 * np.pi) * sigma * energies)
    # energy_weights *= np.exp(-((np.log(energies) - mu) ** 2) / (2 * sigma**2))
    # N = np.sum(energy_weights)
    # energy_weights /= N
    # energy_weights /= energies

    oads = np.array([0.0, 40.0], dtype=np.float32)
    off_axis_softening_fs_interp = np.array([0.0, 0.0], dtype=np.float32)
    mu_w = mu_water(energies)

    phantom = SimplePhantom()
    # phantom.densities[:, 25:71, :] = np.float32(0.2813)  # Feature

    block = Block()
    block.set_square(np.float32(fs))
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
        settings["beam_profile_correction"]["oads"],
        settings["beam_profile_correction"]["factors"],
        k=1,
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
    gpu.d_geo(
        d_geo_grid=d_geo_grid,
        num_voxels=phantom.num_voxels,
        corner=phantom.corner,
        resolution=phantom.resolution,
        source_position=source.position,
    )
    gpu.d_eff(
        d_eff_grid=d_eff_grid,
        num_voxels=phantom.num_voxels,
        corner=phantom.corner,
        resolution=phantom.resolution,
        density_grid=density_grid,
        source_position=source.position,
    )
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
    gpu.mask(
        mask_grid=mask_grid,
        terma_grid=terma_grid,
        num_voxels=phantom.num_voxels,
        resolution=phantom.resolution,
        max_distance_cm=mask_max_distance,
        terma_threshold=mask_terma_threshold * terma_grid.max(),
    )
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
    dose_grid = dose_grid * N * MU * ofc

    # print("Field size: " + str(fs) + "x" + str(fs) + " cm, OFC: " + str(ofc))
    # print("dose_grid: " + str(0.5 * dose_grid[100, 50, 100] + 0.5 * dose_grid[100, 49, 100]))

    return dose_grid


# %%
file_path = "6FFF Beam Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Open Field Depth Dose")
# fig, ax = plt.subplots(1, 1, figsize=(12, 9))
xs = np.linspace(0, 40, 201) + 0.1


def optimise_me(x):
    fss = [3, 4, 6, 8, 10, 20, 30, 40]
    dgs = []
    for fs in fss:
        # print("Optimising for field size: " + str(fs) + " x " + str(fs))
        dose_grid = run(fs, x)
        # ax.plot(xs, dose_grid[100, :, 100], "x")
        dgs.append(dose_grid)

    xs = np.linspace(0, 40, 201) + 0.1
    ofs = [0.8432, 0.8753, 0.9285, 0.9704, 1.000, 1.0837, 1.1190, 1.1349]
    diff = 0.0
    for i in range(len(dgs)):
        calc = dgs[i][100, :, 100]
        gold = np.interp(
            xs,
            df.iloc[6:, 0].to_numpy().astype(np.float32),
            (df.iloc[6:, i + 1] / df.iloc[105, i + 1] * ofs[i] * 0.635)
            .to_numpy()
            .astype(np.float32),
        )
        diff += np.sum(np.abs(calc[6:149] - gold[6:149])) ** 2
    print(f"params: {x}, Diff: {diff}")
    return diff


# %%
def constraint_func(x):
    return x[0] - x[1] - 0.01


# Current best: [0.91712852, 0.88657253]
x0 = [0.90664624, 0.87404703]
bounds = [(0.5, 1.8), (0.5, 1.8)]
# constraints = {
#     "type": "ineq",
#     "fun": constraint_func,
# }


result = minimize(optimise_me, x0, method="Nelder-Mead", bounds=bounds, options={"disp": True})

# # Print the optimization result
# print(f"Optimization Result: {result}")


# %%
fig, ax = plt.subplots(1, 1, figsize=(12, 9))
fss = [3, 4, 6, 8, 10, 20, 30, 40]
# fss = [3, 10, 40]
ofs = [0.8432, 0.8753, 0.9285, 0.9704, 1.000, 1.0837, 1.1190, 1.1349]
dgs = []
for fs in fss:
    print("Optimising for field size: " + str(fs) + " x " + str(fs))
    dose_grid = run(fs, [0.91712852, 0.88657253])
    ax.plot(xs, dose_grid[100, :, 100], "x")
    dgs.append(dose_grid)

# ax.plot(df.iloc[6:, 0], 0.01 * df.iloc[6:, 1], label="3x3")
# ax.plot(df.iloc[6:, 0], 0.01 * df.iloc[6:, 2], label="4x4")
# ax.plot(df.iloc[6:, 0], 0.01 * df.iloc[6:, 3], label="6x6")
# ax.plot(df.iloc[6:, 0], 0.01 * df.iloc[6:, 4], label="8x8")
# ax.plot(df.iloc[6:, 0], 0.01 * df.iloc[6:, 5], label="10x10")
# ax.plot(df.iloc[6:, 0], 0.01 * df.iloc[6:, 6], label="20x20")
# ax.plot(df.iloc[6:, 0], 0.01 * df.iloc[6:, 7], label="30x30")
# ax.plot(df.iloc[6:, 0], 0.01 * df.iloc[6:, 8], label="40x40")
ax.plot(df.iloc[6:, 0], df.iloc[6:, 1] / df.iloc[105, 1] * 0.8432 * 0.635, label="3x3")
ax.plot(df.iloc[6:, 0], df.iloc[6:, 2] / df.iloc[105, 2] * 0.8753 * 0.635, label="4x4")
ax.plot(df.iloc[6:, 0], df.iloc[6:, 3] / df.iloc[105, 3] * 0.9285 * 0.635, label="6x6")
ax.plot(df.iloc[6:, 0], df.iloc[6:, 4] / df.iloc[105, 4] * 0.9704 * 0.635, label="8x8")
ax.plot(df.iloc[6:, 0], df.iloc[6:, 5] / df.iloc[105, 5] * 1.0000 * 0.635, label="10x10")
ax.plot(df.iloc[6:, 0], df.iloc[6:, 6] / df.iloc[105, 6] * 1.0837 * 0.635, label="20x20")
ax.plot(df.iloc[6:, 0], df.iloc[6:, 7] / df.iloc[105, 7] * 1.1190 * 0.635, label="30x30")
ax.plot(df.iloc[6:, 0], df.iloc[6:, 8] / df.iloc[105, 8] * 1.1349 * 0.635, label="40x40")
ax.set_xlim([0, 30])
ax.legend()


# %%
# fmt: off
class Curve:
    def __init__(self, x, y, type, size, of, depth=0.0):
        self.x = x
        self.y = y
        self.type = type
        self.size = size
        self.depth = depth
        self.of = of

df = pd.read_excel(file_path, sheet_name="Open Field Depth Dose")
measured_03x03_pdd = Curve(x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 1]), type="depth", size=3, of=0.8432) 
measured_04x04_pdd = Curve(x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 2]), type="depth", size=4, of=0.8753) 
measured_06x06_pdd = Curve(x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 3]), type="depth", size=6, of=0.9285) 
measured_08x08_pdd = Curve(x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 4]), type="depth", size=8, of=0.9704) 
measured_10x10_pdd = Curve(x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 5]), type="depth", size=10, of=1.000) 
measured_20x20_pdd = Curve(x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 6]), type="depth", size=20, of=1.0837)
measured_30x30_pdd = Curve(x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 7]), type="depth", size=30, of=1.1190)
measured_40x40_pdd = Curve(x=np.array(df.iloc[6:, 0]), y=np.array(df.iloc[6:, 8]), type="depth", size=40, of=1.1349)

df = pd.read_excel(file_path, sheet_name="Open Field Profiles at 1.5cm")
measured_03x03_015 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 1]) * 0.998, type="profile", size=3, of=0.8432, depth=1.5)
measured_04x04_015 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 2]) * 0.998, type="profile", size=4, of=0.8753, depth=1.5)
measured_06x06_015 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 3]) * 0.999, type="profile", size=6, of=0.9285, depth=1.5)
measured_08x08_015 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 4]) * 0.999, type="profile", size=8, of=0.9704, depth=1.5)
measured_10x10_015 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 5]) * 0.999, type="profile", size=10, of=1.000, depth=1.5)
measured_20x20_015 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 6]) * 0.999, type="profile", size=20, of=1.0837, depth=1.5)
measured_30x30_015 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 7]) * 0.997, type="profile", size=30, of=1.1190, depth=1.5)
measured_40x40_015 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 8]) * 0.997, type="profile", size=40, of=1.1349, depth=1.5)

df = pd.read_excel(file_path, sheet_name="Open Field Profiles at 5cm")
measured_03x03_050 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 1]) * 0.804, type="profile", size=3, of=0.8432, depth=5.0)
measured_04x04_050 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 2]) * 0.815, type="profile", size=4, of=0.8753, depth=5.0)
measured_06x06_050 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 3]) * 0.830, type="profile", size=6, of=0.9285, depth=5.0)
measured_08x08_050 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 4]) * 0.841, type="profile", size=8, of=0.9704, depth=5.0)
measured_10x10_050 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 5]) * 0.846, type="profile", size=10, of=1.000, depth=5.0)
measured_20x20_050 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 6]) * 0.859, type="profile", size=20, of=1.0837, depth=5.0)
measured_30x30_050 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 7]) * 0.864, type="profile", size=30, of=1.1190, depth=5.0)
measured_40x40_050 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 8]) * 0.865, type="profile", size=40, of=1.1349, depth=5.0)

df = pd.read_excel(file_path, sheet_name="Open Field Profiles at 10cm")
measured_03x03_100 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 1]) * 0.570, type="profile", size=3, of=0.8432, depth=10.0)
measured_04x04_100 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 2]) * 0.584, type="profile", size=4, of=0.8753, depth=10.0)
measured_06x06_100 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 3]) * 0.605, type="profile", size=6, of=0.9285, depth=10.0)
measured_08x08_100 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 4]) * 0.623, type="profile", size=8, of=0.9704, depth=10.0)
measured_10x10_100 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 5]) * 0.635, type="profile", size=10, of=1.000, depth=10.0)
measured_20x20_100 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 6]) * 0.665, type="profile", size=20, of=1.0837, depth=10.0)
measured_30x30_100 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 7]) * 0.677, type="profile", size=30, of=1.1190, depth=10.0)
measured_40x40_100 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 8]) * 0.682, type="profile", size=40, of=1.1349, depth=10.0)

df = pd.read_excel(file_path, sheet_name="Open Field Profiles at 20cm")
measured_03x03_200 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 1]) * 0.292, type="profile", size=3, of=0.8432, depth=20.0)
measured_04x04_200 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 2]) * 0.301, type="profile", size=4, of=0.8753, depth=20.0)
measured_06x06_200 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 3]) * 0.319, type="profile", size=6, of=0.9285, depth=20.0)
measured_08x08_200 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 4]) * 0.335, type="profile", size=8, of=0.9704, depth=20.0)
measured_10x10_200 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 5]) * 0.347, type="profile", size=10, of=1.000, depth=20.0)
measured_20x20_200 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 6]) * 0.384, type="profile", size=20, of=1.0837, depth=20.0)
measured_30x30_200 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 7]) * 0.400, type="profile", size=30, of=1.1190, depth=20.0)
measured_40x40_200 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 8]) * 0.406, type="profile", size=40, of=1.1349, depth=20.0)

df = pd.read_excel(file_path, sheet_name="Open Field Profiles at 30cm")
measured_03x03_300 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 1]) * 0.156, type="profile", size=3, of=0.8432, depth=30.0)
measured_04x04_300 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 2]) * 0.162, type="profile", size=4, of=0.8753, depth=30.0)
measured_06x06_300 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 3]) * 0.173, type="profile", size=6, of=0.9285, depth=30.0)
measured_08x08_300 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 4]) * 0.184, type="profile", size=8, of=0.9704, depth=30.0)
measured_10x10_300 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 5]) * 0.192, type="profile", size=10, of=1.000, depth=30.0)
measured_20x20_300 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 6]) * 0.221, type="profile", size=20, of=1.0837, depth=30.0)
measured_30x30_300 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 7]) * 0.235, type="profile", size=30, of=1.1190, depth=30.0)
measured_40x40_300 = Curve(x=np.array(df.iloc[8:, 0]), y=np.array(df.iloc[8:, 8]) * 0.241, type="profile", size=40, of=1.1349, depth=30.0)



# fmt: on
# %%
