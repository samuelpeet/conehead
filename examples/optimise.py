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
from conehead.calculate import calculate_fluence, calculate_dose
import pandas as pd
from scipy.optimize import minimize

toml_file = "Truebeam_6_M120.toml"
gold_data_file = "6MV Beam Data.xlsx"


def import_gold_beam_data():
    # Read gold beam PDDs and crossline profiles. Profiles are scaled by the PDD at the profile depth.
    gold = {}
    df = pd.read_excel(gold_data_file, sheet_name="Open Field Profiles at 1.5cm")
    fss = [3, 4, 6, 8, 10, 20, 30, 40]
    # ofs = [0.838663, 0.874284, 0.92817, 0.969307, 1.0, 1.083025, 1.113093, 1.132091]  # 6FFF
    ofs = [0.827548, 0.862568, 0.917877, 0.965234, 1.0, 1.102564, 1.144229, 1.178926]  # 6MV
    for i, v in enumerate(fss):
        gold[v] = {}
        gold[v]["of"] = ofs[i]
        df = pd.read_excel(gold_data_file, sheet_name="Open Field Depth Dose")
        gold[v]["pdd"] = [
            df.iloc[5:, 0].to_numpy().astype(np.float32),
            df.iloc[5:, i + 1].to_numpy().astype(np.float32) / 100,
        ]
        df = pd.read_excel(gold_data_file, sheet_name="Open Field Profiles at 1.5cm")
        gold[v]["prof_15"] = [
            df.iloc[7:, 0].to_numpy().astype(np.float32),
            df.iloc[7:, i + 1].to_numpy().astype(np.float32) / 100 * gold[v]["pdd"][1][15],
        ]
        df = pd.read_excel(gold_data_file, sheet_name="Open Field Profiles at 5cm")
        gold[v]["prof_50"] = [
            df.iloc[7:, 0].to_numpy().astype(np.float32),
            df.iloc[7:, i + 1].to_numpy().astype(np.float32) / 100 * gold[v]["pdd"][1][50],
        ]
        df = pd.read_excel(gold_data_file, sheet_name="Open Field Profiles at 10cm")
        gold[v]["prof_100"] = [
            df.iloc[7:, 0].to_numpy().astype(np.float32),
            df.iloc[7:, i + 1].to_numpy().astype(np.float32) / 100 * gold[v]["pdd"][1][100],
        ]
        df = pd.read_excel(gold_data_file, sheet_name="Open Field Profiles at 20cm")
        gold[v]["prof_200"] = [
            df.iloc[7:, 0].to_numpy().astype(np.float32),
            df.iloc[7:, i + 1].to_numpy().astype(np.float32) / 100 * gold[v]["pdd"][1][200],
        ]
        df = pd.read_excel(gold_data_file, sheet_name="Open Field Profiles at 30cm")
        gold[v]["prof_300"] = [
            df.iloc[7:, 0].to_numpy().astype(np.float32),
            df.iloc[7:, i + 1].to_numpy().astype(np.float32) / 100 * gold[v]["pdd"][1][300],
        ]
    df = pd.read_excel(gold_data_file, sheet_name="Diagonal Profiles")
    gold[40]["diag_15"] = [
        df.iloc[6:, 0].to_numpy().astype(np.float32),
        df.iloc[6:, 1].to_numpy().astype(np.float32) / 100 * gold[40]["pdd"][1][15],
    ]
    gold[40]["diag_15"][0] = gold[40]["diag_15"][0][~np.isnan(gold[40]["diag_15"][1])]
    gold[40]["diag_15"][1] = gold[40]["diag_15"][1][~np.isnan(gold[40]["diag_15"][1])]
    gold[40]["diag_50"] = [
        df.iloc[6:, 0].to_numpy().astype(np.float32),
        df.iloc[6:, 2].to_numpy().astype(np.float32) / 100 * gold[40]["pdd"][1][50],
    ]
    gold[40]["diag_50"][0] = gold[40]["diag_50"][0][~np.isnan(gold[40]["diag_50"][1])]
    gold[40]["diag_50"][1] = gold[40]["diag_50"][1][~np.isnan(gold[40]["diag_50"][1])]
    gold[40]["diag_100"] = [
        df.iloc[6:, 0].to_numpy().astype(np.float32),
        df.iloc[6:, 3].to_numpy().astype(np.float32) / 100 * gold[40]["pdd"][1][100],
    ]
    gold[40]["diag_100"][0] = gold[40]["diag_100"][0][~np.isnan(gold[40]["diag_100"][1])]
    gold[40]["diag_100"][1] = gold[40]["diag_100"][1][~np.isnan(gold[40]["diag_100"][1])]
    gold[40]["diag_200"] = [
        df.iloc[6:, 0].to_numpy().astype(np.float32),
        df.iloc[6:, 4].to_numpy().astype(np.float32) / 100 * gold[40]["pdd"][1][200],
    ]
    gold[40]["diag_200"][0] = gold[40]["diag_200"][0][~np.isnan(gold[40]["diag_200"][1])]
    gold[40]["diag_200"][1] = gold[40]["diag_200"][1][~np.isnan(gold[40]["diag_200"][1])]
    gold[40]["diag_300"] = [
        df.iloc[6:, 0].to_numpy().astype(np.float32),
        df.iloc[6:, 5].to_numpy().astype(np.float32) / 100 * gold[40]["pdd"][1][300],
    ]
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
    fluence_grid = calculate_fluence(
        grid,
        source,
        fluence_map_pri,
        fluence_map_sec,
        settings
    ) 

    dose.values += calculate_dose(
        grid,
        source,
        exam,
        fluence_grid,
        [-fs / 2, fs / 2],
        [-fs / 2, fs / 2],
        settings,
    )
    return dose


def calculate_doses(fss, exam, settings):
    # Calculate conehead doses for a range of field sizes
    grid = Grid(
        corner=np.array([-28.1, 0, -28.1], dtype=np.float32),
        resolution=np.array([0.2, 0.2, 0.2], dtype=np.float32),
        num_voxels=np.array([281, 281, 281], dtype=np.int32),
    )
    calc = {}
    for fs in fss:
        # print(f"Running field: {fs}")
        calc[fs] = {}
        calc[fs]["dose"] = run(fs, grid, exam, settings)
        calc[fs][
            "dose"
        ].values *= 100  # Multiplying by 100 for 100 MU (thus 1 Gy at dmax for 10x10)
        ds = [
            grid.corner[1] + grid.resolution[1] / 2 + x * grid.resolution[1]
            for x in range(grid.num_voxels[1])
        ]
        calc[fs]["pdd"] = [
            ds,
            calc[fs]["dose"].values[grid.num_voxels[2] // 2, :, grid.num_voxels[0] // 2],
        ]
        xs = [
            grid.corner[0] + grid.resolution[0] / 2 + x * grid.resolution[0]
            for x in range(grid.num_voxels[0])
        ]
        calc[fs]["prof_15"] = [xs, calc[fs]["dose"].values[grid.num_voxels[2] // 2, 7, :]]
        calc[fs]["prof_50"] = [
            xs,
            0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 24, :]
            + 0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 25, :],
        ]
        calc[fs]["prof_100"] = [
            xs,
            0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 49, :]
            + 0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 50, :],
        ]
        calc[fs]["prof_200"] = [
            xs,
            0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 99, :]
            + 0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 100, :],
        ]
        calc[fs]["prof_300"] = [
            xs,
            0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 149, :]
            + 0.5 * calc[fs]["dose"].values[grid.num_voxels[2] // 2, 150, :],
        ]
        if fs == 40:
            xs = np.array(xs)
            rs = np.sqrt(xs * xs + xs * xs)
            rs[xs < 0] *= -1
            calc[fs]["diag_15"] = [rs, np.diag(calc[fs]["dose"].values[:, 7, :])]
            calc[fs]["diag_50"] = [
                rs,
                0.5 * np.diag(calc[fs]["dose"].values[:, 24, :])
                + 0.5 * np.diag(calc[fs]["dose"].values[:, 25, :]),
            ]
            calc[fs]["diag_100"] = [
                rs,
                0.5 * np.diag(calc[fs]["dose"].values[:, 49, :])
                + 0.5 * np.diag(calc[fs]["dose"].values[:, 50, :]),
            ]
            calc[fs]["diag_200"] = [
                rs,
                0.5 * np.diag(calc[fs]["dose"].values[:, 99, :])
                + 0.5 * np.diag(calc[fs]["dose"].values[:, 100, :]),
            ]
            calc[fs]["diag_300"] = [
                rs,
                0.5 * np.diag(calc[fs]["dose"].values[:, 149, :])
                + 0.5 * np.diag(calc[fs]["dose"].values[:, 150, :]),
            ]
    return calc


def difference_in_pdds(calc, gold):
    diff = 0.0
    for fs in calc.keys():
        calc_pdd_interp = np.interp(
            gold[fs]["pdd"][0], calc[fs]["pdd"][0], calc[fs]["pdd"][1] / calc[fs]["pdd"][1].max()
        )
        # calc_pdd_interp = calc_pdd_interp / calc_pdd_interp[100] 
        gold_pdd = gold[fs]["pdd"][1] / gold[fs]["pdd"][1].max()

        # Clip off approximate build-up region
        calc_pdd_interp = calc_pdd_interp[10:]
        gold_pdd = gold_pdd[10:]

        diff += np.sum(np.abs(calc_pdd_interp - gold_pdd) ** 2)
    return diff


def optimise_pdd(x, exam, gold):
    # Update settings with new energy spectrum
    settings = toml.load(toml_file)
    energies = np.array(settings["energy_spectrum"]["energies"], dtype=np.float32)

    c1 = x[0]
    c2 = x[1]
    energy_weights = energies ** (c1) * np.exp(-c2 * energies)
    N = np.sum(energy_weights)
    energy_weights /= N
    energy_weights /= energies
    settings["energy_spectrum"]["weights_3"] = energy_weights.tolist()
    settings["energy_spectrum"]["weights_10"] = energy_weights.tolist()
    settings["energy_spectrum"]["weights_40"] = energy_weights.tolist()


    # mu = x[0]
    # sigma = x[1]
    # energy_weights = 1 / (np.sqrt(2 * np.pi) * sigma * energies)
    # energy_weights *= np.exp(-((-np.log(energies) - mu) ** 2) / (2 * sigma**2))
    
    # energy_weights = np.array(
    #     [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]], dtype=np.float32
    # )
    # N = np.sum(energy_weights)
    # energy_weights /= N
    # energy_weights *= energies
    # settings["energy_spectrum"]["weights_3"] = energy_weights.tolist()

    # Calc doses
    fss = [40]
    calc = calculate_doses(fss, exam, settings)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    for fs in fss:
        ax.plot(gold[fs]["pdd"][0], gold[fs]["pdd"][1] / gold[fs]["pdd"][1].max(), label="Measured PDD")
        ax.plot(
            calc[fs]["pdd"][0], calc[fs]["pdd"][1] / calc[fs]["pdd"][1].max(), ".", label="Conehead PDD"
        )
        ax.set_xlabel("Depth (cm)")
        ax.set_ylabel("Relative Dose")
        ax.legend()
    plt.show()
    diff = difference_in_pdds(calc, gold)
    print(f"params: {x}\nDiff: {diff*100}")
    return diff * 100


# %%
# Optimising energy spectrum
gold = import_gold_beam_data()
exam = Exam(dicom_dir="40", hu_lut_path="Siemens_Confidence.toml")
# x0 = [0.49161678, 0.36570732, 0.32814098, 0.1115883, 0.46065874, 0.19569992, 0.13531406, 0.03368567, 0.02094162, 0.00338178, 0.01224467, 0.00159998]
# bounds = [
#     (0.0, 1.0),
#     (0.0, 1.0),
#     (0.0, 1.0),
#     (0.0, 1.0),
#     (0.0, 1.0),
#     (0.0, 1.0),
#     (0.0, 1.0),
#     (0.0, 1.0),
#     (0.0, 1.0),
#     (0.0, 1.0),
#     (0.0, 1.0),
#     (0.0, 1.0)
# ]
# x0 = [2.35476884, 1.3858963] # 3 x 3
# x0 = [1.33441245, 0.97771809]  # 10 x 10
x0 = [2.46958337, 1.7451692]  # 40 x 40
bounds = [(0.01, 2.5), (0.01, 2.5)]
result = minimize(optimise_pdd, x0, args=(exam, gold), method="Nelder-Mead", bounds=bounds)

# %%
# Plot PDDs
# The following are for 6FFF
# x_3 = [9.78887112e-01, 5.06918312e-03, 1.80647223e-03, 1.35677832e-04, 4.02326888e-02, 2.48797532e-02, 5.64411258e-05, 4.50911833e-05, 2.54285319e-03, 3.81893355e-04, 9.19434324e-06, 2.01442085e-05]
# x_10 = [6.37866626e-01, 5.24930987e-02, 1.02095518e-01, 3.23278437e-05, 2.71260518e-02, 8.53287051e-03, 4.25837066e-05, 0.00000000e+00, 1.22616270e-03, 6.78531340e-04, 2.59861780e-04, 3.32664982e-05]
# x_40 = [3.11421911e-01, 6.85822548e-02, 2.12540862e-01, 3.23181680e-05, 5.48393311e-03, 1.93255940e-03, 3.55081544e-05, 0.00000000e+00, 2.96541822e-04, 2.79249071e-06, 3.87204832e-04, 2.59697551e-05]
# The following are for 6X
x_3 = [8.96620071e-01, 4.20301952e-01, 3.29980288e-01, 1.97611172e-01, 1.05119587e-01, 6.81861541e-02, 3.61792269e-02, 1.24138023e-02, 6.23102064e-03, 5.78580583e-04, 1.89360807e-03, 1.31881297e-03]
x_10 = [8.81758150e-01, 5.02334426e-01, 2.82445606e-01, 1.39655376e-01, 6.96135187e-02, 3.08778158e-02, 1.84210722e-02, 8.34655595e-03, 5.03643674e-03, 3.13199028e-03, 1.87479388e-03, 8.49120953e-04]
x_40 = [9.48665723e-01, 6.04128249e-01, 4.35059848e-01, 1.57176706e-01, 6.19487082e-02, 2.27400882e-02, 7.83719644e-03, 3.56620846e-03, 1.90168948e-03, 6.54339377e-04, 1.08546203e-03, 1.93403650e-04]
x = x_3

settings = toml.load(toml_file)
energies = np.array(settings["energy_spectrum"]["energies"], dtype=np.float32)

energy_weights = np.array(
    [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]], dtype=np.float32
)
# mu = x[0]
# sigma = x[1]
# energy_weights = 1 / (np.sqrt(2 * np.pi) * sigma * energies)
# energy_weights *= np.exp(-((np.log(energies) - mu) ** 2) / (2 * sigma**2))
# e_ave = x[0]
# e_mp = x[1]
# c1 = e_ave / (e_ave - e_mp)
# c2 = 1 / (e_ave - e_mp)
# c1 = x[0]
# c2 = x[1]
# energy_weights = energies ** (c1) * np.exp(-c2 * energies)
# energy_weights *= energies

N = np.sum(energy_weights)
energy_weights /= N
energy_weights *= energies

settings["energy_spectrum"]["weights_10"] = energy_weights.tolist()
fss = [3, 10, 40]
calc = calculate_doses(fss, exam, settings)
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
for fs in fss:
    ax.plot(gold[fs]["pdd"][0], gold[fs]["pdd"][1]/gold[fs]["pdd"][1].max(), label="Measured PDD")
    ax.plot(
        calc[fs]["pdd"][0], calc[fs]["pdd"][1] / calc[fs]["pdd"][1].max(), ".", label="Conehead PDD"
    )
    ax.set_xlabel("Depth (cm)")
    ax.set_ylabel("Relative Dose")
    ax.legend()


# %%
def calculate_normalisation():
    exam = Exam(dicom_dir="40", hu_lut_path="Siemens_Confidence.toml")
    settings = toml.load(toml_file)
    settings["calculation"]["normalisation"] = np.float32(1.0)
    calc = calculate_doses([10], exam, settings)
    d_max = np.argmax(calc[10]["pdd"][1])
    calc_dose_at_dmax = calc[10]["pdd"][1][d_max]
    normalisation = 1 / calc_dose_at_dmax
    return normalisation

print(calculate_normalisation())


# %%
def calculate_ofcs(fss, exam, settings, gold):
    # Set OFCs to 1
    fss = settings["output_factor_correction"]["field_sizes"]
    settings["output_factor_correction"]["factors"] = np.ones_like(fss, dtype=np.float32)
    calc = calculate_doses(fss, exam, settings)
    for fs in fss:
        # gold_dose_at_10cm = gold[fs]["of"] * 0.635  # 0.635 is PDD(10) for 10 x 10 field (6FFF)
        gold_dose_at_10cm = gold[fs]["of"] * 0.664  # 0.664 is PDD(10) for 10 x 10 field (6X)
        calc_dose_at_10cm = 0.5 * calc[fs]["pdd"][1][49] + 0.5 * calc[fs]["pdd"][1][50]
        ofc = gold_dose_at_10cm / calc_dose_at_10cm
        calc[fs]["ofc"] = ofc
    return calc

gold = import_gold_beam_data()
exam = Exam(dicom_dir="40", hu_lut_path="Siemens_Confidence.toml")
settings = toml.load(toml_file)
fss = [3, 4, 6, 8, 10, 20, 30, 40]
calc = calculate_ofcs(fss, exam, settings, gold)
for fs in calc.keys():
    calc[fs]["ofc"] /= calc[10]["ofc"]  # Normalise to 10 x 10
print("Output Factor Corrections:")
print([calc[fs]["ofc"] for fs in fss])


# %%
def difference_in_diag(calc, gold):
    diff = 0.0
    fs = 40
    key = "diag_15"
    calc_diag_interp = np.interp(
        gold[fs][key][0], calc[fs][key][0], calc[fs][key][1] / calc[fs][key][1].max()
    )
    gold_diag = gold[fs][key][1] / gold[fs][key][1].max()

    # Clip off tails
    calc_diag_interp = calc_diag_interp[50:-50]
    gold_diag = gold_diag[50:-50]

    diff += np.sum(np.abs(calc_diag_interp - gold_diag) ** 2)
    return diff

def optimise_bpc(x, exam, gold):
    # Update settings with new beam profile correction
    settings = toml.load(toml_file)
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
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
result = minimize(optimise_bpc, x0, args=(exam, gold), method="Nelder-Mead", bounds=bounds)




















# %%
def difference_in_penumbras(calc, gold):
    diff = 0.0
    key = "prof_15"

    # fs = 3
    # calc_prof_interp = np.interp(
    #     gold[fs][key][0], calc[fs][key][0], calc[fs][key][1] / calc[fs][key][1].max()
    # )
    # gold_prof = gold[fs][key][1] / gold[fs][key][1][253]
    # # Isolate penumbra of 3cm field
    # calc_prof_interp = calc_prof_interp[253:293]
    # gold_prof = gold_prof[253:293]
    # diff += np.sum(np.abs(calc_prof_interp - gold_prof) ** 2)
    # print("3cm diff: ", np.sum(np.abs(calc_prof_interp - gold_prof) ** 2))

    fs = 20
    calc_prof_interp = np.interp(
        gold[fs][key][0], calc[fs][key][0], calc[fs][key][1] / calc[fs][key][1][140]
    )
    gold_prof = gold[fs][key][1] / gold[fs][key][1][253]
    # Isolate penumbra of 20cm field
    calc_prof_interp = calc_prof_interp[333:383]
    gold_prof = gold_prof[333:383]
    diff += np.sum(np.abs(calc_prof_interp - gold_prof) ** 2)
    print("20cm diff: ", np.sum(np.abs(calc_prof_interp - gold_prof) ** 2))
    # fs = 40
    # calc_prof_interp = np.interp(
    #     gold[fs][key][0], calc[fs][key][0], calc[fs][key][1] / calc[fs][key][1].max()
    # )
    # gold_prof = gold[fs][key][1] / gold[fs][key][1][253]
    # # Isolate penumbra of 40cm field
    # calc_prof_interp = calc_prof_interp[433:483]
    # gold_prof = gold_prof[433:483]
    # diff += np.sum(np.abs(calc_prof_interp - gold_prof) ** 2)
    # print("40cm diff: ", np.sum(np.abs(calc_prof_interp - gold_prof) ** 2))

    return diff


def optimise_penumbras(x, exam, gold):
    # Update settings with new beam profile correction
    settings = toml.load(toml_file)
    settings["sources"]["pri_s"] = x[0]
    settings["sources"]["pri_x"] = x[1]
    settings["sources"]["pri_y"] = x[1]
    settings["sources"]["sec_s"] = 1 - x[0]
    settings["sources"]["sec_x"] = x[2]
    settings["sources"]["sec_y"] = x[2]
    settings["sources"]["sec_z"] = x[3]
    calc = calculate_doses([20], exam, settings)
    diff = difference_in_penumbras(calc, gold) * 100
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    # for i, fs in enumerate([3, 20, 40]):
    for i, fs in enumerate([20]):
        ax.plot(gold[fs]["prof_15"][0], gold[fs]["prof_15"][1]/gold[fs]["prof_15"][1][253] + i*0.1, color='red', label='Measured')
        ax.plot(calc[fs]["prof_15"][0], calc[fs]["prof_15"][1]/calc[fs]["prof_15"][1][140] + i*0.1, '-', color="#3535ff", label='Calculated')
    ax.axis([0, 15, 0, 1.14])
    ax.grid()
    plt.show()
    print(f"Params: {x}")
    print(f"Diff: {diff}")
    return diff


# Optimising penumbras
gold = import_gold_beam_data()
exam = Exam(dicom_dir="40", hu_lut_path="Siemens_Confidence.toml")
# Params: [ 0.80255114  0.20141664 15.41493598  1.69573431]
# x0 = [0.93, 0.05, 1.8, 5]
x0 = [0.92,  0.19827905, 4.29646987,  0.92904012]
bounds = [(0.75, 1.0), (0.01, 0.3), (1.0, 15.0), (0.5, 10)]
result = minimize(optimise_penumbras, x0, args=(exam, gold), method="Nelder-Mead", bounds=bounds)


























# %%
settings = toml.load(toml_file)
exam = Exam(dicom_dir="40", hu_lut_path="Siemens_Confidence.toml")
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
# gold = import_gold_beam_data()
for i in [(0, 0.5), (1, 1.0), (3, 2.0), (7, 4.0), (11, 6.0)]:
    energy_weights = np.zeros(12, dtype=np.float32)
    energy_weights[i[0]] = 1.0
    settings["energy_spectrum"]["weights"] = energy_weights.tolist()
    fs = 20
    calc = calculate_doses([fs], exam, settings)
    # ax.plot(gold[fs]["pdd"][0], gold[fs]["pdd"][1] / gold[fs]["pdd"][1][100], label="Measured PDD")

    ax.plot(
        calc[fs]["pdd"][0], calc[fs]["pdd"][1] / calc[fs]["pdd"][1].max(), label=f"{i[1]} MeV"
    )
ax.legend()
ax.grid()

# %%
# Profile and PDD agreement plots
gold = import_gold_beam_data()
exam = Exam(dicom_dir="40", hu_lut_path="Siemens_Confidence.toml")
settings = toml.load(toml_file)
# fss = [3, 4, 6, 8, 10, 20, 30, 40]
fss = [20]
calc = calculate_doses(fss, exam, settings)

#%%
# pdd10 = 0.635  # For 6FFF
pdd10 = 0.664  # For 6X
plt.style.use('default')
for fs in fss:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Top subplot: overlay both curves
    ax1.set_title("Percentage Depth Dose Comparison for Field Size: {} cm".format(fs))
    ax1.plot(gold[fs]["pdd"][0], gold[fs]["pdd"][1]/gold[fs]["pdd"][1][100]*gold[fs]["of"] * pdd10, label="Measured")
    ax1.plot(calc[fs]["pdd"][0], calc[fs]["pdd"][1], 'o', color='red', fillstyle='none', label="Conehead")  # Unfilled red circles
    ax1.set_ylabel("Dose (Gy)")
    ax1.legend()
    ax1.grid(which='both', linewidth=0.5, alpha=0.7)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    
    # Set major ticks and gridlines every 0.1 Gy on the y-axis
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax1.grid(which='major', linewidth=1.0)  # Major gridlines bolder
    ax1.grid(which='minor', linewidth=0.5)  # Minor gridlines thinner
    
    # Bottom subplot: percentage difference
    gold_dose = gold[fs]["pdd"][1]/gold[fs]["pdd"][1][100]*gold[fs]["of"] * pdd10
    calc_dose_interp = np.interp(gold[fs]["pdd"][0], calc[fs]["pdd"][0], calc[fs]["pdd"][1])
    pct_diff = 100 * (calc_dose_interp - gold_dose) / gold_dose
    abs_diff = calc_dose_interp - gold_dose
    ax2.plot(gold[fs]["pdd"][0], abs_diff, color='black', linewidth=2)  # Thick black line
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(-0.02, 0.02)
    ax2.set_xlabel("Depth (cm)")
    ax2.set_ylabel("Absolute Difference (Gy)")
    ax2.grid(which='both', linewidth=0.5, alpha=0.7)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax2.grid(which='major', linewidth=1.0)  # Major gridlines bolder
    ax2.grid(which='minor', linewidth=0.5)  # Minor gridlines thinner

    plt.tight_layout()

# %%
# pdd10 = 0.635  # For 6FFF
pdd10 = 0.664  # For 6X
plt.style.use('dark_background')
lines = []
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
for fs in fss:
    # line1, = ax.plot(gold[fs]["pdd"][0], gold[fs]["pdd"][1]/gold[fs]["pdd"][1][100]*gold[fs]["of"] * 0.664, color='red', label='Measured')
    line1, = ax.plot(gold[fs]["pdd"][0], gold[fs]["pdd"][1]/gold[fs]["pdd"][1][100]*gold[fs]["of"] * pdd10, color='red', label='Measured')
    line2, = ax.plot(calc[fs]["pdd"][0], calc[fs]["pdd"][1], '-', color='#3535ff', label='Calculated')          
    lines.append(line1)
    lines.append(line2)
    ax.set_facecolor("#1f1f1f")
    fig.patch.set_facecolor('#1f1f1f')
    ax.grid(which='major', linewidth=0.5, alpha=0.7, color="#cccccc")
    ax.grid(which='minor', linewidth=0.5, alpha=0.7, color="#cccccc", linestyle='dotted')
    ax.xaxis.set_major_locator(plt.MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.set_ylabel("Dose (Gy)")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Depth (cm)")
    ax.set_title("PDD Comparison for Various Field Sizes (3, 4, 6, 8, 10, 20, 30, 40 cm)")
    ax.spines['top'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    ax.spines['right'].set_color('#cccccc')
    ax.tick_params(axis='both', colors='#cccccc', width=0.5)
    plt.tight_layout()

plt.legend(handles=[lines[0], lines[1]])

# %%
# pdd10 = 0.635  # For 6FFF
pdd10 = 0.664  # For 6X
plt.style.use('dark_background')
lines = []
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
for fs in fss:
    line1, = ax.plot(gold[fs]["prof_15"][0], gold[fs]["prof_15"][1]/gold[fs]["pdd"][1][100]*gold[fs]["of"] * pdd10, color='red', label='Measured')
    line2, = ax.plot(calc[fs]["prof_15"][0], calc[fs]["prof_15"][1], '-', color="#3535ff", label='Calculated')
    ax.plot(gold[fs]["prof_50"][0], gold[fs]["prof_50"][1]/gold[fs]["pdd"][1][100]*gold[fs]["of"] * pdd10, color='red', label='Measured')
    ax.plot(calc[fs]["prof_50"][0], calc[fs]["prof_50"][1], '-', color='#3535ff', label='Calculated')    
    ax.plot(gold[fs]["prof_100"][0], gold[fs]["prof_100"][1]/gold[fs]["pdd"][1][100]*gold[fs]["of"] * pdd10, color='red', label='Measured')
    ax.plot(calc[fs]["prof_100"][0], calc[fs]["prof_100"][1], '-', color='#3535ff', label='Calculated')   
    ax.plot(gold[fs]["prof_200"][0], gold[fs]["prof_200"][1]/gold[fs]["pdd"][1][100]*gold[fs]["of"] * pdd10, color='red', label='Measured')
    ax.plot(calc[fs]["prof_200"][0], calc[fs]["prof_200"][1], '-', color='#3535ff', label='Calculated')   
    ax.plot(gold[fs]["prof_300"][0], gold[fs]["prof_300"][1]/gold[fs]["pdd"][1][100]*gold[fs]["of"] * pdd10, color='red', label='Measured')
    ax.plot(calc[fs]["prof_300"][0], calc[fs]["prof_300"][1], '-', color='#3535ff', label='Calculated')               
    lines.append(line1)
    lines.append(line2)
    ax.set_facecolor("#1f1f1f")
    fig.patch.set_facecolor('#1f1f1f')
    ax.grid(which='major', linewidth=0.5, alpha=0.7, color="#cccccc")
    ax.grid(which='minor', linewidth=0.5, alpha=0.7, color="#cccccc", linestyle='dotted')
    ax.xaxis.set_major_locator(plt.MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.spines['top'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    ax.spines['right'].set_color('#cccccc')
    ax.tick_params(axis='both', colors='#cccccc', width=0.5)
    ax.set_ylabel("Dose (Gy)")
    ax.set_xlim(-30, 30)
    ax.set_xlabel("Position (cm)")
    ax.set_title("Lateral Profile Comparison for Various Field Sizes (3, 4, 6, 8, 10, 20, 30, 40 cm)")
    plt.tight_layout()    
plt.legend(handles=[lines[0], lines[1]])

# %%
# pdd10 = 0.635  # For 6FFF
pdd10 = 0.664  # For 6X
plt.style.use('dark_background')
lines = []
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fs = 40
line1, = ax.plot(gold[fs]["diag_15"][0], gold[fs]["diag_15"][1]/gold[fs]["pdd"][1][100]*gold[fs]["of"] * pdd10, color='red', label='Measured')
line2, = ax.plot(calc[fs]["diag_15"][0], calc[fs]["diag_15"][1], '-', color="#3535ff", label='Calculated')
ax.plot(gold[fs]["diag_50"][0], gold[fs]["diag_50"][1]/gold[fs]["pdd"][1][100]*gold[fs]["of"] * pdd10, color='red', label='Measured')
ax.plot(calc[fs]["diag_50"][0], calc[fs]["diag_50"][1], '-', color='#3535ff', label='Calculated')    
ax.plot(gold[fs]["diag_100"][0], gold[fs]["diag_100"][1]/gold[fs]["pdd"][1][100]*gold[fs]["of"] * pdd10, color='red', label='Measured')
ax.plot(calc[fs]["diag_100"][0], calc[fs]["diag_100"][1], '-', color='#3535ff', label='Calculated')   
ax.plot(gold[fs]["diag_200"][0], gold[fs]["diag_200"][1]/gold[fs]["pdd"][1][100]*gold[fs]["of"] * pdd10, color='red', label='Measured')
ax.plot(calc[fs]["diag_200"][0], calc[fs]["diag_200"][1], '-', color='#3535ff', label='Calculated')   
ax.plot(gold[fs]["diag_300"][0], gold[fs]["diag_300"][1]/gold[fs]["pdd"][1][100]*gold[fs]["of"] * pdd10, color='red', label='Measured')
ax.plot(calc[fs]["diag_300"][0], calc[fs]["diag_300"][1], '-', color='#3535ff', label='Calculated')               
lines.append(line1)
lines.append(line2)
ax.set_facecolor("#1f1f1f")
fig.patch.set_facecolor('#1f1f1f')
ax.grid(which='major', linewidth=0.5, alpha=0.7, color="#cccccc")
ax.grid(which='minor', linewidth=0.5, alpha=0.7, color="#cccccc", linestyle='dotted')
ax.xaxis.set_major_locator(plt.MultipleLocator(5.0))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1.0))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
ax.spines['top'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')
ax.spines['left'].set_color('#cccccc')
ax.spines['right'].set_color('#cccccc')
ax.tick_params(axis='both', colors='#cccccc', width=0.5)
ax.set_ylabel("Dose (Gy)")
ax.set_xlim(-35, 35)
ax.set_xlabel("Position (cm)")
ax.set_title("Diagonal Profile Comparison for 40 x 40 cm2 Field Size")
plt.tight_layout()
plt.legend(handles=[lines[0], lines[1]])

# %%
