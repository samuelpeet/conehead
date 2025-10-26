import numpy as np
import numpy.typing as npt


class SimplePhantom:
    def __init__(self):
        # Create simple 40 cm cube water phantom in DICOM coords

        # self.positions: npt.NDArray[np.float32] = np.mgrid[-20:20:41j, -40:0:41j, -20:20:41j].astype(np.float32)
        # _, xlen, ylen, zlen = self.positions.shape
        self.num_voxels: npt.NDArray[np.int32] = np.array([201, 201, 201], dtype=np.int32)
        self.corner: npt.NDArray[np.float32] = np.array([-20.1, -20.1, -20.1], dtype=np.float32)
        self.resolution: npt.NDArray[np.float32] = np.array([0.2, 0.2, 0.2], dtype=np.float32)
        self.densities: npt.NDArray[np.float32] = np.ones(
            (self.num_voxels[2], self.num_voxels[1], self.num_voxels[0]), dtype=np.float32
        )  # Water
        self.densities[80:121, 80:121, 80:121] = np.float32(4)  # Higher density feature
        # phantom_densities[15:26, 15:26, 15:26] = np.float32(4)  # Higher density feature
