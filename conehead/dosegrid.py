import numpy as np
import numpy.typing as npt


class DoseGrid:

    def __init__(self, size: npt.NDArray[np.int32], origin: npt.NDArray[np.float32], spacing: npt.NDArray[np.float32]):
        self.num_voxels: npt.NDArray[np.int32] = size
        self.corner: npt.NDArray[np.float32] = origin
        self.resolution: npt.NDArray[np.float32] = spacing
        self.dose: npt.NDArray[np.float32] = np.zeros(self.num_voxels, dtype=np.float32)
