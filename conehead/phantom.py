import numpy as np
import numpy.typing as npt


class SimplePhantom:

    def __init__(self):
        # Create simple 40 cm cube water phantom in global coords
        self.positions: npt.NDArray[np.float32] = np.mgrid[-20:20:41j, -20:20:41j, -40:0:41j].astype(np.float32)
        _, xlen, ylen, zlen = self.positions.shape
        self.densities: npt.NDArray[np.float32] = np.ones((xlen, ylen, zlen), dtype=np.float32)  # Water
        # phantom_densities[15:26, 15:26, 15:26] = np.float32(4)  # Higher density feature
