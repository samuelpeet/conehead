import numpy as np
import numpy.typing as npt


class DoseGrid:
    def __init__(
        self,
        num_voxels: npt.ArrayLike,
        corner: npt.ArrayLike,
        resolution: npt.ArrayLike,
    ):
        self.num_voxels: npt.NDArray[np.int32] = np.asarray(num_voxels, dtype=np.int32)
        self.corner: npt.NDArray[np.float32] = np.asarray(corner, dtype=np.float32)
        self.resolution: npt.NDArray[np.float32] = np.asarray(resolution, dtype=np.float32)
        self.dose: npt.NDArray[np.float32] = np.zeros(
            (self.num_voxels[2], self.num_voxels[1], self.num_voxels[0]), dtype=np.float32
        )
