import numpy as np


class Source:

    def __init__(self):
        self._sad = 1000
        self._position = np.array([0, 0, self._sad])
        self._rotation = np.array([0, 0, 0])
        self._block_plane = np.zeros((400, 400))

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
    def block_plane(self):
        return self._block_plane
    
    @block_plane.setter
    def block_plane(self, new_block_plane):
        self._block_plane = new_block_plane

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
        x = self._sad * np.cos(phi * np.pi / 180)
        y = self.position[1]
        z = self._sad * np.sin(phi * np.pi / 180)
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