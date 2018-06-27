import numpy as np
from numpy.linalg import inv


class Source:

    def __init__(self):
        self._sad = 1000
        self._position = np.array([0, self._sad, 0])
        self._rotation = np.array([0, 0, np.pi])

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

    def gantry(self, theta):
        """ Set the gantry angle of the source.

        Parameters
        ----------
        theta : float
            The gantry angle in degrees. Must be within the range [0, 360).
        """
        assert theta >= 0 and theta < 360, "Invalid gantry angle"

        # Set new source position
        theta = (90 - theta) % 360  # From IEC 61217 to global
        x = self._sad * np.cos(theta * np.pi / 180)
        y = self._sad * np.sin(theta * np.pi / 180)
        z = self.position[2]
        self.position = np.array([x, y, z])

        # Set new source rotation
        rx = self.rotation[0]
        ry = self.rotation[1]
        rz = (90 + theta) % 360 * np.pi / 180
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
        ry = (360 - theta) * np.pi / 180  # From IEC 61217 to global
        rz = self.rotation[2]
        self.rotation = np.array([rx, ry, rz])


def beam_to_global(beam_coords, source_position, source_rotation):
    """Transform from beam coordinates to global coordinates.

    Computes coordinate transformation:

    G = Rz * Ry * B + S

    where B is the point in beam coordinates, Rz and Ry are rotation matrices
    about their respective (global) axes, and S is the location of the source
    in global coordinates.
    
    Parameters
    ----------
    beam_coords : ndarray
        The coordinates in the beam geometry to be transformed.
    source_position : ndarray
        The position of the source in global coordinates
    source_position : ndarray
        The rotation of the source in global coordinates

    Returns
    -------
    ndarray
        The beam coordinates transformed into the global geometry.
    """
    (rx, ry, rz) = source_rotation
    # (cx, sx) = (np.cos(rx), np.sin(rx))
    (cy, sy) = (np.cos(ry), np.sin(ry))
    (cz, sz) = (np.cos(rz), np.sin(rz))

    # Rotate about y-axis (collimator)
    rot_y_matrix = np.array([[cy, 0, sy],
                             [0, 1, 0],
                             [-sy, 0, cy]])
    global_coords = np.matmul(rot_y_matrix, beam_coords)

    # Rotate about z-axis (gantry)
    rot_z_matrix = np.array([[cz, -sz, 0],
                             [sz, cz, 0],
                             [0, 0, 1]])
    global_coords = np.matmul(rot_z_matrix, global_coords)

    # Perform translation
    global_coords = source_position + global_coords

    return global_coords


def global_to_beam(global_coords, source_position, source_rotation):
    """Transform from global coordinates to beam coordinates.

    Computes coordinate transformation:

    B = inv(Rz) * inv(Rz) * (B - S)

    where G is the point in global coordinates, Rz and Ry are rotation matrices
    about their respective (global) axes, and S is the location of the source
    in global coordinates.

    Parameters
    ----------
    global_coords : ndarray
        The coordinates in the global geometry to be transformed.
    source_position : ndarray
        The position of the source in global coordinates
    source_position : ndarray
        The rotation of the source in global coordinates

    Returns
    -------
    ndarray
        The global coordinates transformed into the beam geometry.
    """
    (rx, ry, rz) = source_rotation
    # (cx, sx) = (np.cos(rx), np.sin(rx))
    (cy, sy) = (np.cos(ry), np.sin(ry))
    (cz, sz) = (np.cos(rz), np.sin(rz))

    # Perform translation
    beam_coords = global_coords - source_position

    # Rotate about z-axis (gantry)
    rot_z_matrix = np.array([[cz, -sz, 0],
                             [sz, cz, 0],
                             [0, 0, 1]])
    beam_coords = np.matmul(inv(rot_z_matrix), beam_coords)

    # Rotate about y-axis (collimator)
    rot_y_matrix = np.array([[cy, 0, sy],
                             [0, 1, 0],
                             [-sy, 0, cy]])
    beam_coords = np.matmul(inv(rot_y_matrix), beam_coords)

    return beam_coords
