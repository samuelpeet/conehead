import numpy as np
from numpy.linalg import inv


class Transformer:

    def __init__(self, source_position, source_rotation):
        """Enable transormation between global and beam coordinates. 
    
        Expensive operations such as inversion of rotation matrices are calculated 
        only once upon instatiation.
        
        Parameters
        ----------
        source_position : ndarray
            The position of the source in global coordinates
        source_position : ndarray
            The rotation of the source in global coordinates
        
        """
        self.source_position = source_position
        self.source_rotation = source_rotation

        (_, ry, rz) = source_rotation
        (cy, sy) = (np.cos(-ry), np.sin(-ry))
        (cz, sz) = (np.cos(rz), np.sin(rz))

        self.rot_y_matrix = np.array([[cy, 0, sy],
                                      [0, 1, 0],
                                      [-sy, 0, cy]])
        self.inv_rot_y_matrix = inv(self.rot_y_matrix)

        self.rot_z_matrix = np.array([[cz, -sz, 0],
                                      [sz, cz, 0],
                                      [0, 0, 1]])
        self.inv_rot_z_matrix = inv(self.rot_z_matrix)

    def beam_to_global(self, beam_coords):
        """Transform from beam coordinates to global coordinates.

        Compute coordinate transformation:

        G = Ry * Rz * B + S

        where B is the point in beam coordinates, Ry and Rz are rotation
        matrices about their respective (global) axes, and S is the location of
        the source in global coordinates.

        Parameters
        ----------
        beam_coords : ndarray
            The coordinates in the beam geometry to be transformed.

        Returns
        -------
        ndarray
            The beam coordinates transformed into the global geometry.
        """
        # Rotate about z-axis (collimator)
        global_coords = np.matmul(self.rot_z_matrix, beam_coords)

        # Rotate about y-axis (gantry)
        global_coords = np.matmul(self.rot_y_matrix, global_coords)

        # Perform translation
        global_coords = self.source_position + global_coords

        return global_coords


    def global_to_beam(self, global_coords):
        """Transform from global coordinates to beam coordinates.

        Computes coordinate transformation:

        B = inv(Rz) * inv(Ry) * (B - S)

        where G is the point in global coordinates, Rz and Ry are rotation
        matrices about their respective (global) axes, and S is the location of
        the source in global coordinates.

        Parameters
        ----------
        global_coords : ndarray
            The coordinates in the global geometry to be transformed.

        Returns
        -------
        ndarray
            The global coordinates transformed into the beam geometry.
        """
        # Perform translation
        beam_coords = global_coords - self.source_position

        # Rotate about y-axis (gantry)
        beam_coords = np.matmul(self.inv_rot_y_matrix, beam_coords)

        # Rotate about z-axis (collimator)
        beam_coords = np.matmul(self.inv_rot_z_matrix, beam_coords)

        return beam_coords


def line_block_plane_collision(ray_direction, epsilon=1e-6):
    """ Calculate the point of intersection of a line and the blocking plane.

     Parameters
    ----------
    ray_direction : ndarray
        Direction vector of ray, normalisation not necessary
    epsilon : float
        Cutoff to determine if ray intersects with plane

    Returns
    -------
    ndarray
        Coordinates of line plane intersection
    """
    plane_normal = np.array([0, 0, 1])  # Always towards source
    plane_point = plane_point = np.array([0, 0, -100])  # Isocentre
    ray_point = np.array([0, 0, 0])  # Source position

    ndotu = plane_normal.dot(ray_direction)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")
    w = ray_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    psi = w + si * ray_direction + plane_point
    return psi


def line_calc_limit_plane_collision(ray_direction, plane_point, epsilon=1e-6):
    """ Calculate the point of intersection of a line and the calculation
    limit plane.

     Parameters
    ----------
    ray_direction : ndarray
        Direction vector of ray, normalisation not necessary
    plane_point : ndarray
        A point lying on the calculation limit plane 
    epsilon : float
        Cutoff to determine if ray intersects with plane

    Returns
    -------
    ndarray
        Coordinates of line plane intersection
    """
    plane_normal = np.array([0, 0, 1])  # Always towards source
    plane_point = plane_point
    ray_point = np.array([0, 0, 0])  # Source position

    ndotu = plane_normal.dot(ray_direction)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")
    w = ray_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    psi = w + si * ray_direction + plane_point
    return psi

def isocentre_plane_position(position, SAD):
    """Calculate the position of a voxel projected to the isocentre plane.
    
    Parameters
    ----------
    position : ndarray
        Position of voxel in beam coordinates
    SAD : float
        Source axis distance

    Returns
    ------
    ndarray
        Voxel position projected to the isocentre plane
    """
    x, y, z = position
    t_iso = SAD / z
    x_iso = t_iso * x
    y_iso = t_iso * y
    return np.array([x_iso, y_iso])