import numpy as np
from numpy.linalg import inv


def beam_to_global(beam_coords, source_position, source_rotation):
    """Transform from beam coordinates to global coordinates.

    Compute coordinate transformation:

    G = Ry * Rz * B + S

    where B is the point in beam coordinates, Ry and Rz are rotation matrices
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
    (_, ry, rz) = source_rotation
    # (cx, sx) = (np.cos(rx), np.sin(rx))
    (cy, sy) = (np.cos(-ry), np.sin(-ry))
    (cz, sz) = (np.cos(rz), np.sin(rz))

    # Rotate about z-axis (collimator)
    rot_z_matrix = np.array([[cz, -sz, 0],
                             [sz, cz, 0],
                             [0, 0, 1]])
    global_coords = np.matmul(rot_z_matrix, beam_coords)

    # Rotate about y-axis (gantry)
    rot_y_matrix = np.array([[cy, 0, sy],
                             [0, 1, 0],
                             [-sy, 0, cy]])
    global_coords = np.matmul(rot_y_matrix, global_coords)

    # Perform translation
    global_coords = source_position + global_coords

    return global_coords


def global_to_beam(global_coords, source_position, source_rotation):
    """Transform from global coordinates to beam coordinates.

    Computes coordinate transformation:

    B = inv(Rz) * inv(Ry) * (B - S)

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
    (_, ry, rz) = source_rotation
    # (cx, sx) = (np.cos(rx), np.sin(rx))
    (cy, sy) = (np.cos(-ry), np.sin(-ry))
    (cz, sz) = (np.cos(rz), np.sin(rz))

    # Perform translation
    beam_coords = global_coords - source_position

    # Rotate about y-axis (gantry)
    rot_y_matrix = np.array([[cy, 0, sy],
                             [0, 1, 0],
                             [-sy, 0, cy]])
    beam_coords = np.matmul(inv(rot_y_matrix), beam_coords)

    # Rotate about z-axis (collimator)
    rot_z_matrix = np.array([[cz, -sz, 0],
                             [sz, cz, 0],
                             [0, 0, 1]])
    beam_coords = np.matmul(inv(rot_z_matrix), beam_coords)

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


def line_calc_limit_plane_collision(ray_direction, epsilon=1e-6):
    """ Calculate the point of intersection of a line and the calculation
    limit plane.

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
    plane_point = np.array([0, 0, -50])  # Half iso for now
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