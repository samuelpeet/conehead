from conehead.source import Source
from conehead.block import Block
from conehead.phantom import SimplePhantom
from conehead.conehead import Conehead

import numpy as np
import pydicom
from scipy.interpolate import RegularGridInterpolator

# Load test plan
plan = pydicom.dcmread("RP.3DCRT.dcm", force=True)

# Choose source
source = Source("varian_clinac_6MV")
source.gantry = 0
source.collimator = 45

# Set the jaws and MLC
# block = Block(source.rotation, plan=plan)
block = Block()
block.set_square(10)

# Use a simple cubic phantom
phantom = SimplePhantom()

# Calculation settings
settings = {
    # 'stepSize': 0.1,  # Stepsize when raytracing effective depth (cm)
    'sPri': 0.90924,  # Primary source strength
    'sAnn': 2.887e-3,  # Annular source strength
    'zAnn': 4.0,  # Distance of annular source from point source (cm)
    'rInner': 0.2,  # Inner radius of annular source
    'rOuter': 1.4,  # Outer radius of annular source
    'zExp': 12.5,  # Distance of exponential source from point source (cm)
    'sExp': 8.289e-3,  # Exponential source strength
    'kExp': 0.4816,  # Exponential source exponent coefficient
    'softRatio': 0.0025,  # cm^-1
    'softLimit': 20,  # cm
    'hornRatio': 0.0065,  # % per cm
    'eLow': 0.01,  # MeV
    'eHigh': 7.0,  # MeV
    'eNum': 500,  # Spectrum samples
    'fluenceResampling': 3  # Split voxels for fluence calculation
}

conehead = Conehead()
conehead.calculate(source, block, phantom, settings)
# conehead.plot()

# import cProfile
# cProfile.run('conehead.calculate(source, block, phantom, settings)')