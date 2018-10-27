# kernel.py
#
# Energy deposition kernel data for varian 6MV source created with EDKnrc.
#
# Simulation details:
#
# Histories - 20000000
# Num cones - 48
# Angles - 3.75 degree spacing
# Num spheres - 24
# Medium - H20521ICRU
# ECUT - 0.521
# PCUT - 0.010
#
import numpy as np
import pandas as pd


class PolyenergeticKernel:

    def __init__(self):

        # Differential kernel data
        df = pd.read_csv("kerneldata/varian_6.csv")
        differential = {}
        differential["radii"] = df["Radius"].unique()
        for angle in df["Angle"].unique():
            values = df.loc[df["Angle"] == angle]["Value"].tolist()
            differential["{:.2f}".format(180 - angle)] = values

        # Normalise kernel data
        f_total = 0.289622754141
        for key in differential.keys():
            if key != "radii":
                differential[key] = np.array(differential[key]) / f_total
        self.differential = differential

        # Calculate cumulative kernel data
        cumulative = {}
        cumulative["radii"] = differential["radii"]
        for key in differential.keys():
            if key != "radii":
                cumulative[key] = np.cumsum(differential[key])
                cumulative[key] = np.interp(  # Resample to 0.1 mm
                    np.linspace(0.05, 60.0, 5996),
                    cumulative["radii"],
                    cumulative[key]
                )
        self.cumulative = cumulative
