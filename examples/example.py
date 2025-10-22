from conehead.source import Source
from conehead.block import Block
from conehead.phantom import SimplePhantom
from conehead.conehead import Conehead

import numpy as np
import pydicom
from scipy.interpolate import RegularGridInterpolator
import toml

# Load test plan
plan = pydicom.dcmread("RP.3DCRT.dcm", force=True)

# Choose source
source = Source("varian_clinac_6MV")
source.gantry = 0
source.collimator = 0

# Set the jaws and MLC
# block = Block(source.rotation, plan=plan)
block = Block()
block.set_square(10)

# Use a simple cubic phantom
phantom = SimplePhantom()


# Load calculation settings from TOML file
settings_toml = toml.load("settings.toml")
settings = settings_toml["calculation"]
settings["energy_weights"] = settings_toml["energy_weights"]


conehead = Conehead()
conehead.calculate(source, block, phantom, settings)
# conehead.plot()

# import cProfile
# cProfile.run('conehead.calculate(source, block, phantom, settings)')