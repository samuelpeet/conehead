from conehead import (
    Source, Block , SimplePhantom, Conehead
)
import numpy as np
import pydicom
from scipy.interpolate import RegularGridInterpolator

# Load test plan
plan = pydicom.dcmread("RP.3DCRT.dcm", force=True)

# Choose source
source = Source("varian_clinac_6MV")
source.gantry(0)
source.collimator(0)

# Set the jaws and MLC
# block = Block(source.rotation, plan=plan)
block = Block()
block.set_square(30)

# Use a simple cubic phantom
phantom = SimplePhantom()

# Calculation settings
settings = {
    'stepSize': 0.1,  # Stepsize when raytracing effective depth (cm)
    'sPri': 1.0,  # Primary source strength (photons/cm^2)
    'softRatio': 0.0025,  # cm^-1
    'softLimit': 20,  # cm
    'hornRatio': 0.006,  # % per cm
    'eLow': 0.01,  # MeV
    'eHigh': 7.0,  # MeV
    'eNum': 500,  # Spectrum samples
}

conehead = Conehead()
conehead.calculate(source, block, phantom, settings)
conehead.plot()
