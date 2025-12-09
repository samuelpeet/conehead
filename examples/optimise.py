# %%
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../build/"))
import matplotlib.pyplot as plt
import numpy as np
import toml
from conehead.dicom import export_dose
from conehead.exam import Exam
from conehead.plan import Plan
from conehead.grid import Grid
from conehead.block import Block
from conehead.source import Source
from conehead.calculate import calculate
import pandas as pd
from scipy.optimize import minimize


def import_gold_beam_data():
    # Read gold beam PDDs and crossline profiles. Profiles are scaled by the PDD at the profile depth.
    gold = {}
    df = pd.read_excel("6FFF Beam Data.xlsx", sheet_name="Open Field Profiles at 1.5cm")
    fss = [3, 4, 6, 8, 10, 20, 30, 40]
    ofs = [0.838663, 0.874284, 0.92817, 0.969307, 1.0, 1.083025, 1.113093, 1.132091]
    for i, v in enumerate(fss):
        gold[v] = {}
        gold[v]["of"] = ofs[i]
        df = pd.read_excel("6FFF Beam Data.xlsx", sheet_name="Open Field Depth Dose")
        gold[v]["pdd"] = [df.iloc[5:, 0].to_numpy().astype(np.float32), df.iloc[5:, i + 1].to_numpy().astype(np.float32) / 100]
        df = pd.read_excel("6FFF Beam Data.xlsx", sheet_name="Open Field Profiles at 1.5cm")
        gold[v]["prof_15"] = [df.iloc[7:, 0].to_numpy().astype(np.float32), df.iloc[7:, i + 1].to_numpy().astype(np.float32) / 100 * gold[v]["pdd"][1][15]]
        df = pd.read_excel("6FFF Beam Data.xlsx", sheet_name="Open Field Profiles at 5cm")
        gold[v]["prof_50"] = [df.iloc[7:, 0].to_numpy().astype(np.float32), df.iloc[7:, i + 1].to_numpy().astype(np.float32) / 100 * gold[v]["pdd"][1][50]]    
        df = pd.read_excel("6FFF Beam Data.xlsx", sheet_name="Open Field Profiles at 10cm")
        gold[v]["prof_100"] = [df.iloc[7:, 0].to_numpy().astype(np.float32), df.iloc[7:, i + 1].to_numpy().astype(np.float32) / 100 * gold[v]["pdd"][1][100]]
        df = pd.read_excel("6FFF Beam Data.xlsx", sheet_name="Open Field Profiles at 20cm")
        gold[v]["prof_200"] = [df.iloc[7:, 0].to_numpy().astype(np.float32), df.iloc[7:, i + 1].to_numpy().astype(np.float32) / 100 * gold[v]["pdd"][1][200]]
        df = pd.read_excel("6FFF Beam Data.xlsx", sheet_name="Open Field Profiles at 30cm")
        gold[v]["prof_300"] = [df.iloc[7:, 0].to_numpy().astype(np.float32), df.iloc[7:, i + 1].to_numpy().astype(np.float32) / 100 * gold[v]["pdd"][1][300]]
    df = pd.read_excel("6FFF Beam Data.xlsx", sheet_name="Diagonal Profiles")
    gold[40]["diag_15"] = [df.iloc[6:, 0].to_numpy().astype(np.float32), df.iloc[6:, 1].to_numpy().astype(np.float32) / 100 * gold[40]["pdd"][1][15]]
    gold[40]["diag_15"][0] = gold[40]["diag_15"][0][~np.isnan(gold[40]["diag_15"][1])]
    gold[40]["diag_15"][1] = gold[40]["diag_15"][1][~np.isnan(gold[40]["diag_15"][1])]
    gold[40]["diag_50"] = [df.iloc[6:, 0].to_numpy().astype(np.float32), df.iloc[6:, 2].to_numpy().astype(np.float32) / 100 * gold[40]["pdd"][1][50]]
    gold[40]["diag_50"][0] = gold[40]["diag_50"][0][~np.isnan(gold[40]["diag_50"][1])]
    gold[40]["diag_50"][1] = gold[40]["diag_50"][1][~np.isnan(gold[40]["diag_50"][1])]
    gold[40]["diag_100"] = [df.iloc[6:, 0].to_numpy().astype(np.float32), df.iloc[6:, 3].to_numpy().astype(np.float32) / 100 * gold[40]["pdd"][1][100]]
    gold[40]["diag_100"][0] = gold[40]["diag_100"][0][~np.isnan(gold[40]["diag_100"][1])]
    gold[40]["diag_100"][1] = gold[40]["diag_100"][1][~np.isnan(gold[40]["diag_100"][1])]
    gold[40]["diag_200"] = [df.iloc[6:, 0].to_numpy().astype(np.float32), df.iloc[6:, 4].to_numpy().astype(np.float32) / 100 * gold[40]["pdd"][1][200]]
    gold[40]["diag_200"][0] = gold[40]["diag_200"][0][~np.isnan(gold[40]["diag_200"][1])]
    gold[40]["diag_200"][1] = gold[40]["diag_200"][1][~np.isnan(gold[40]["diag_200"][1])]
    gold[40]["diag_300"] = [df.iloc[6:, 0].to_numpy().astype(np.float32), df.iloc[6:, 5].to_numpy().astype(np.float32) / 100 * gold[40]["pdd"][1][300]]
    gold[40]["diag_300"][0] = gold[40]["diag_300"][0][~np.isnan(gold[40]["diag_300"][1])]
    gold[40]["diag_300"][1] = gold[40]["diag_300"][1][~np.isnan(gold[40]["diag_300"][1])]
    return gold


def run(fs, grid, exam, settings):
    source = Source()
    source.gantry = 0.0
    source.collimator = 90.0
    block = Block(settings=settings)
    block.set_square(fs)
    fluence_map_pri, fluence_map_sec = block.get_fluence_maps()
    dose = Grid(corner=grid.corner, resolution=grid.resolution, num_voxels=grid.num_voxels)
    dose.values += calculate(  # type: ignore
        grid,
        source,
        fluence_map_pri,
        fluence_map_sec,
        exam,
        [-fs / 2, fs / 2],
        [-fs / 2, fs / 2],
        settings,
    )
    return dose


def calculate_doses(fss, exam, settings):
    # Calculate conehead doses for a range of field sizes
    grid = Grid(
        corner=np.array([-30.1, 0, -30.1], dtype=np.float32),
        resolution=np.array([0.2, 0.2, 0.2], dtype=np.float32),
        num_voxels=np.array([301, 201, 301], dtype=np.int32),
    )
    calc = {}
    for fs in fss:
        print(f"Running field: {fs}")
        calc[fs] = {}
        calc[fs]["dose"] = run(fs, grid, exam, settings) 
        calc[fs]["dose"].values *= 100 # Multiplying by 100 for 100 MU (thus 1 Gy at dmax for 10x10)
        ds = [grid.corner[1] + grid.resolution[1] / 2 + x * grid.resolution[1] for x in range(grid.num_voxels[1])]
        calc[fs]["pdd"] = [ds, calc[fs]["dose"].values[grid.num_voxels[1] // 2, :, grid.num_voxels[1] // 2]]
        xs = [grid.corner[0] + grid.resolution[0] / 2 + x * grid.resolution[0] for x in range(grid.num_voxels[0])]
        calc[fs]["prof_15"] = [xs, calc[fs]["dose"].values[grid.num_voxels[2] // 2, 7, :]]
        calc[fs]["prof_50"] = [
            xs,
            0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 24, :] + 
            0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 25, :]
        ]
        calc[fs]["prof_100"] = [
            xs,
            0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 49, :] + 
            0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 50, :]
        ]
        calc[fs]["prof_200"] = [
            xs,
            0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 99, :] + 
            0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 100, :]
        ]
        calc[fs]["prof_300"] = [
            xs,
            0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 149, :] + 
            0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 150, :]
        ]
        if fs == 40:
            xs = np.array(xs)
            rs = np.sqrt(xs * xs + xs * xs)
            rs[xs < 1] *= -1
            calc[fs]["diag_15"] = [rs, np.diag(calc[fs]["dose"].values[:, 7, :])]
            calc[fs]["diag_50"] = [
                rs,
                0.5 * np.diag(calc[fs]["dose"].values[:, 24, :]) +
                0.5 * np.diag(calc[fs]["dose"].values[:, 25, :])
            ]
            calc[fs]["diag_100"] = [
                rs,
                0.5 * np.diag(calc[fs]["dose"].values[:, 49, :]) +
                0.5 * np.diag(calc[fs]["dose"].values[:, 50, :])
            ]
            calc[fs]["diag_200"] = [
                rs,
                0.5 * np.diag(calc[fs]["dose"].values[:, 99, :]) +
                0.5 * np.diag(calc[fs]["dose"].values[:, 100, :])
            ]
            calc[fs]["diag_300"] = [
                rs,
                0.5 * np.diag(calc[fs]["dose"].values[:, 149, :]) +
                0.5 * np.diag(calc[fs]["dose"].values[:, 150, :])
            ]
    return calc


def difference_in_pdds(calc, gold):
    diff = 0.0
    for fs in calc.keys():
        calc_pdd_interp = np.interp(gold[fs]["pdd"][0], calc[fs]["pdd"][0], calc[fs]["pdd"][1] / calc[fs]["pdd"][1].max())
        gold_pdd = gold[fs]["pdd"][1]

        # Clip off approximate build-up region
        calc_pdd_interp = calc_pdd_interp[10:]
        gold_pdd = gold_pdd[10:]

        diff += np.sum(np.abs(calc_pdd_interp - gold_pdd)**2)
    return diff


def optimise_pdd(x, exam, gold):
    # Update settings with new energy spectrum
    settings = toml.load("Truebeam_6FFF_M120.toml")
    energies = np.array(settings["energy_spectrum"]["energies"], dtype=np.float32)
    energy_weights = np.array([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 0.0, x[8], 0.0, x[9]], dtype=np.float32)
    N = np.sum(energy_weights)
    energy_weights /= N
    energy_weights /= energies
    settings["energy_spectrum"]["weights"] = energy_weights.tolist()

    # Calc doses
    fss = [3, 10, 40]
    calc = calculate_doses(fss, exam, settings)
    diff = difference_in_pdds(calc, gold)
    print(f"params: {x}, Diff: {diff}")
    return diff


# %%
# Optimising energy spectrum
gold = import_gold_beam_data()
exam = Exam(dicom_dir="40", hu_lut_path="Siemens_Confidence.toml")
x0 = [0.13077393, 0.08785954, 0.0294463, 0.04815249, 0.03111974, 0.43969824, 0.03215145, 0.0169554, 0.03014422, 0.022631]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
          (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
          (0.0, 1.0), (0.0, 1.0)]
result = minimize(optimise_pdd, x0, args=(exam, gold), method="Nelder-Mead", bounds=bounds)

# %%
# Plot PDDs
# x = [0.11815628, 0.07121678, 0.14556541, 0.09402617, 0.10339564, 0.09395615, 0.07655023, 0.11007001, 0.0512248, 0.0117276]
x = [0.13077393, 0.08785954, 0.0294463, 0.04815249, 0.03111974, 0.43969824, 0.03215145, 0.0169554, 0.03014422, 0.022631]
settings = toml.load("Truebeam_6FFF_M120.toml")
energies = np.array(settings["energy_spectrum"]["energies"], dtype=np.float32)
energy_weights = np.array([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 0.0, x[8], 0.0, x[9]], dtype=np.float32)
N = np.sum(energy_weights)
energy_weights /= N
energy_weights /= energies
settings["energy_spectrum"]["weights"] = energy_weights.tolist()
fss = [3, 10, 40]
calc = calculate_doses(fss, exam, settings)
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
for fs in fss:
    ax.plot(gold[fs]["pdd"][0], gold[fs]["pdd"][1], label="Measured PDD")
    ax.plot(calc[fs]["pdd"][0], calc[fs]["pdd"][1] / calc[fs]["pdd"][1].max(), '.', label="Conehead PDD")
    ax.set_xlabel("Depth (cm)")
    ax.set_ylabel("Relative Dose")
    ax.legend()

# %%
def calculate_normalisation():
    exam = Exam(dicom_dir="40", hu_lut_path="Siemens_Confidence.toml")
    settings = toml.load("Truebeam_6FFF_M120.toml")
    settings["calculation"]["normalisation"] = np.float32(1.0)
    calc = calculate_doses([10], exam, settings)
    d_max = np.argmax(calc[10]["pdd"][1])
    calc_dose_at_10cm = calc[10]["pdd"][1][d_max]
    normalisation = 1 / calc_dose_at_10cm
    return normalisation
print(calculate_normalisation())


# %%
def calculate_ofcs(fss, exam, settings, gold):
    # Set OFCs to 1
    fss = settings["output_factor_correction"]["field_sizes"]
    settings["output_factor_correction"]["factors"] = np.ones_like(fss, dtype=np.float32)
    calc = calculate_doses(fss, exam, settings)
    for fs in fss:
        gold_dose_at_10cm = gold[fs]["of"] * 0.635  # 0.635 is PDD(10) for 10 x 10 field
        calc_dose_at_10cm = 0.5 * calc[fs]["pdd"][1][49] + 0.5 * calc[fs]["pdd"][1][50]
        ofc = gold_dose_at_10cm / calc_dose_at_10cm
        calc[fs]["ofc"] = ofc
    return calc

gold = import_gold_beam_data()
exam = Exam(dicom_dir="40", hu_lut_path="Siemens_Confidence.toml")
settings = toml.load("Truebeam_6FFF_M120.toml")
fss = [3, 4, 6, 8, 10, 20, 30, 40]
calc = calculate_ofcs(fss, exam, settings, gold)
for fs in calc.keys():
    calc[fs]["ofc"] /= calc[10]["ofc"] # Normalise to 10 x 10
print("Output Factor Corrections:")
print([calc[fs]["ofc"] for fs in fss])


# %%
def difference_in_diag(calc, gold):
    diff = 0.0
    fs = 40
    key = "diag_15"
    calc_diag_interp = np.interp(gold[fs][key][0], calc[fs][key][0], calc[fs][key][1] / calc[fs][key][1].max())
    gold_diag = gold[fs][key][1]

    # Clip off tails
    calc_diag_interp = calc_diag_interp[50:-50]
    gold_diag = gold_diag[50:-50]

    diff += np.sum(np.abs(calc_diag_interp - gold_diag)**2)
    return diff

def optimise_bpc(x, exam, gold):
    # Update settings with new beam profile correction
    settings = toml.load("Truebeam_6FFF_M120.toml")
    settings["beam_profile_correction"]["factors"] = np.insert(x, 0, 1.00)
    calc = calculate_doses([40], exam, settings)
    diff = difference_in_diag(calc, gold) * 100
    print(f"Params: {x}")
    print(f"Diff: {diff}")
    return diff

# Optimising beam profile correction
gold = import_gold_beam_data()
exam = Exam(dicom_dir="40", hu_lut_path="Siemens_Confidence.toml")
x0 = [0.996, 0.982, 0.961, 0.936, 0.881, 0.822, 0.765, 0.720, 0.677, 0.631, 0.596, 0.562, 0.533, 0.51, 0.5, 0.388, 0.069, 0.043, 0.03, 0.02, 0.02, 0.02]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
          (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
          (0.0, 1.0), (0.0, 1.0)]
result = minimize(optimise_bpc, x0, args=(exam, gold), method="Nelder-Mead", bounds=bounds)


# %%
