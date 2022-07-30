import numpy as np
import numpy.typing as npt


class Source:

    def __init__(self, model: str, sad: np.float32 = np.float32(100)):
        self._sad: np.float32 = sad
        self._position: npt.NDArray[np.float32] = np.array([0, 0, self._sad], dtype=np.float32)
        self._rotation: npt.NDArray[np.float32] = np.array([0, 0, 0], dtype=np.float32)

        if model == "varian_clinac_6MV":
            import conehead.varian_clinac_6MV
            self.weights = conehead.varian_clinac_6MV.weights_ali
        else:
            raise NotImplementedError("The requested model is not yet"
                                      " implemented.")

    @property
    def position(self) -> npt.NDArray[np.float32]:
        return self._position

    @position.setter
    def position(self, new_postion: npt.NDArray[np.float32]):
        self._position: npt.NDArray[np.float32] = new_postion

    @property
    def rotation(self) -> npt.NDArray[np.float32]: 
        return self._rotation

    @rotation.setter
    def rotation(self, new_rotation: npt.NDArray[np.float32]):
        self._rotation: npt.NDArray[np.float32] = new_rotation

    @property
    def sad(self) -> np.float32:
        return self._sad

    def gantry(self, theta: np.float32):
        """ Set the gantry angle of the source.

        Parameters
        ----------
        theta : float
            The gantry angle in degrees. Must be within the range [0, 360).
        """
        assert theta >= 0 and theta < 360, "Invalid gantry angle"

        # Set new source position
        phi: np.float32 = (90 - theta) % 360  # IEC 61217
        x: np.float32 = self.sad * np.cos(phi * np.pi / 180)
        y: np.float32 = self.position[1]
        z: np.float32 = self.sad * np.sin(phi * np.pi / 180)
        self.position = np.array([x, y, z])

        # Set new source rotation
        rx: np.float32 = self.rotation[0]
        ry: np.float32 = (0 - theta) % 360 * np.pi / 180
        rz: np.float32 = self.rotation[2]
        self.rotation = np.array([rx, ry, rz], dtype=np.float32)

    def collimator(self, theta: np.float32):
        """ Set the collimator angle of the source.

        Parameters
        ----------
        theta : float
            The collimator angle in degrees. Must be within the range [0, 360).
        """
        assert theta >= 0 and theta < 360, "Invalid collimator angle"

        rx: np.float32 = self.rotation[0]
        ry: np.float32 = self.rotation[1]
        rz: np.float32 = theta * np.pi / 180
        self.rotation = np.array([rx, ry, rz], dtype=np.float32)
