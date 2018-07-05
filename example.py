from conehead.source import Source
from conehead.block import Block
from conehead.phantom import Phantom
from conehead.conehead import Conehead


# Create source
source = Source("varian_clinac_6MV")
source.gantry(0)
source.collimator(0)

# Create 10 cm x 10 cm collimator opening
block = Block(source.rotation)
block.set_square(10)

# Create simple phantom
phantom = Phantom()

# Calculation settings
settings = {
    'stepSize': 0.1,  # Stepsize when raytracing effective depth
    'sPri': 1.0,  # Primary source strength (photons/mm^2)
    'softRatio': 0.0025,  # mm^-1
    'softLimit': 20,  # cm
    'eLow': 0.01,  # MeV
    'eHigh': 7.0,  # MeV
    'eNum': 500,  # Spectrum samples
}

calculation = Conehead()
calculation.calculate(source, block, phantom, settings)
calculation.plot()
