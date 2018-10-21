import numpy as np


class Source:

    def __init__(self, source, SAD=100):
        self._SAD = SAD
        self._position = np.array([0, 0, self._SAD])
        self._rotation = np.array([0, 0, 0])

        if source == "varian_clinac_6MV":
            import conehead.varian_clinac_6MV
            self.weights = conehead.varian_clinac_6MV.weights_ali
        else:
            raise NotImplementedError("The requested source is not yet"
                                      " implemented.")

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_postion):
        self._position = new_postion

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, new_rotation):
        self._rotation = new_rotation

    @property
    def SAD(self):
        return self._SAD

    def gantry(self, theta):
        """ Set the gantry angle of the source.

        Parameters
        ----------
        theta : float
            The gantry angle in degrees. Must be within the range [0, 360).
        """
        assert theta >= 0 and theta < 360, "Invalid gantry angle"

        # Set new source position
        phi = (90 - theta) % 360  # IEC 61217
        x = self._SAD * np.cos(phi * np.pi / 180)
        y = self.position[1]
        z = self._SAD * np.sin(phi * np.pi / 180)
        self.position = np.array([x, y, z])

        # Set new source rotation
        rx = self.rotation[0]
        ry = (0 - theta) % 360 * np.pi / 180
        rz = self.rotation[2]
        self.rotation = np.array([rx, ry, rz])

    def collimator(self, theta):
        """ Set the collimator angle of the source.

        Parameters
        ----------
        theta : float
            The collimator angle in degrees. Must be within the range [0, 360).
        """
        assert theta >= 0 and theta < 360, "Invalid collimator angle"

        rx = self.rotation[0]
        ry = self.rotation[1]
        rz = theta * np.pi / 180
        self.rotation = np.array([rx, ry, rz])
