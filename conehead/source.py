import numpy as np
import numpy.typing as npt


class Source:

    def __init__(self, model: str, sad: np.float32 = np.float32(100)):

        if model == "varian_clinac_6MV":
            import conehead.varian_clinac_6MV
            self.weights = conehead.varian_clinac_6MV.weights_ali
        else:
            raise NotImplementedError("The requested model is not yet"
                                      " implemented.")

        # Initialize source to gantry and collimator zero
        self._sad: np.float32 = sad
        self._gantry: np.float32 = 0
        self._collimator: np.float32 = 0
        self._position: npt.NDArray[np.float32] = np.array([0, -self._sad, 0], dtype=np.float32)

        # Basis of source local coordinate system
        self.v_x: npt.NDArray[np.float32] = np.array([1, 0, 0], dtype=np.float32)
        self.v_y: npt.NDArray[np.float32] = np.array([0, 1, 0], dtype=np.float32)
        self.v_z: npt.NDArray[np.float32] = np.array([0, 0, 1], dtype=np.float32)
        
        # Create rotation matrix to transform from world coords to local source coords
        self.transform: npt.NDArray[np.float32] = np.array([self.v_x, self.v_y, self.v_z]).transpose()
        self.transform = np.linalg.inv(self.transform)

    @property
    def position(self) -> npt.NDArray[np.float32]:
        return self._position

    @position.setter
    def position(self, new_postion: npt.NDArray[np.float32]):
        self._position: npt.NDArray[np.float32] = new_postion

    @property
    def sad(self) -> np.float32:
        return self._sad

    @property
    def gantry(self) -> npt.NDArray[np.float32]: 
        return self._gantry

    @gantry.setter
    def gantry(self, theta: np.float32):
        """ Set the gantry angle of the source.

        Parameters
        ----------
        theta : float
            The gantry angle in degrees. Must be within the range [0, 360).
        """
        assert theta >= 0 and theta < 360, "Invalid gantry angle"
        self._gantry: npt.NDArray[np.float32] = theta
        self._update_geometry()

    @property
    def collimator(self) -> npt.NDArray[np.float32]: 
        return self._collimator

    @collimator.setter
    def collimator(self, theta: np.float32):
        """ Set the collimator angle of the source.

        Parameters
        ----------
        theta : float
            The collimator angle in degrees. Must be within the range [0, 360).
        """
        assert theta >= 0 and theta < 360, "Invalid collimator angle"
        self._collimator: npt.NDArray[np.float32] = theta
        self._update_geometry()

    def _update_geometry(self):

        # Set new source position
        theta = self._gantry
        phi: np.float32 = (90 - theta) % 360  # IEC 61217
        x: np.float32 = self.sad * np.cos(phi * np.pi / 180)
        y: np.float32 = self.sad * -np.sin(phi * np.pi / 180)
        z: np.float32 = self.position[2]
        self.position = np.array([x, y, z])

        # Construct new basis
        v_x: npt.NDArray[np.float32] = np.array([1, 0, 0], dtype=np.float32)
        v_y: npt.NDArray[np.float32] = np.array([0, 1, 0], dtype=np.float32)
        v_z: npt.NDArray[np.float32] = np.array([0, 0, 1], dtype=np.float32)

        # Rotate basis with new collimator angle
        t = -self._collimator * np.pi / 180
        r_y = np.array([[np.cos(t), 0, np.sin(t)],
                        [0, 1, 0],
                        [-np.sin(t), 0, np.cos(t)]])
        new_v_x = np.matmul(r_y, v_x)
        new_v_z = np.matmul(r_y, v_z)
        
        # Rotate basis with new gantry angle
        p = self._gantry * np.pi / 180
        r_z = np.array([[np.cos(p), -np.sin(p), 0],
                        [np.sin(p), np.cos(p), 0],
                        [0, 0, 1]])
        self.v_x = np.matmul(r_z, new_v_x)
        self.v_y = np.matmul(r_z, v_y)
        self.v_z = np.matmul(r_z, new_v_z)

        # Update transformation matrix
        self.transform: npt.NDArray[np.float32] = np.array([self.v_x, self.v_y, self.v_z]).transpose()
        self.transform = np.linalg.inv(self.transform)
