import numpy as np
import numpy.typing as npt


class DoseGrid:

    def __init__(self, size: list[int], origin: npt.NDArray[np.float32], spacing: npt.NDArray[np.float32]):
        self.size: list[int] = size
        self.origin: npt.NDArray[np.float32] = origin
        self.spacing: npt.NDArray[np.float32] = spacing
        self.dose: npt.NDArray[np.float32] = np.zeros(self.size, dtype=np.float32)
