import numpy as np
import numpy.typing as npt


class SimplePhantom:

    def __init__(self):
        # Create simple 40 cm cube water phantom in DICOM coords

        # self.positions: npt.NDArray[np.float32] = np.mgrid[-20:20:41j, -40:0:41j, -20:20:41j].astype(np.float32)
        # _, xlen, ylen, zlen = self.positions.shape
        self.num_voxels: list[int] = [102, 102, 102]
        self.corner: npt.NDArray[np.float32] = np.array([-10.1, 0, -10.1], dtype=np.float32)
        self.resolution: npt.NDArray[np.float32] = np.array([.2, .2, .2], dtype=np.float32)
        self.densities: npt.NDArray[np.float32] = np.ones(self.size, dtype=np.float32)  # Water
        # phantom_densities[15:26, 15:26, 15:26] = np.float32(4)  # Higher density feature
