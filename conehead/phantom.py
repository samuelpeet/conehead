import numpy as np


class SimplePhantom:

    def __init__(self):
        # Create simple 40 cm cube water phantom in global coords
        self.positions = np.mgrid[-20:20:41j, -20:20:41j, -30:10:41j]
        _, xlen, ylen, zlen = self.positions.shape
        self.densities = np.ones((xlen, ylen, zlen))  # Water
        # phantom_densities[15:26, 15:26, 15:26] = 4  # Higher density feature
