import numpy as np
import cython
cimport numpy as cnp

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def dda_3d_c(cnp.float64_t[:] direction, cnp.int32_t[:] grid_shape, cnp.int32_t[:] first_voxel, cnp.float64_t[:] voxel_size):
    """ Calculate the intersection points of a ray with a voxel grid, using a
    3D DDA algorithm.

    Parameters
    ----------
    direction : ndarray
        direction vector of the ray
    grid : ndarray
        Shape of 3D voxel grid
    first_voxel : ndarray
        Index of ray source voxel
    voxel_size : ndarray
        Size of voxel dimensions

    Returns
    -------
    intersection_t_values_mv : ndarray
        Array of t values corresponding to voxel boundary intersections
    voxels_traversed_mv : ndarray
        List of voxel indices intersected by ray
    """
    cdef cnp.int32_t step[3]
    step[0] = -1 if direction[0] < 0 else 1
    step[1] = -1 if direction[1] < 0 else 1
    step[2] = -1 if direction[2] < 0 else 1
    
    cdef cnp.int32_t current_voxel[3]
    current_voxel[0] = first_voxel[0]
    current_voxel[1] = first_voxel[1]
    current_voxel[2] = first_voxel[2]

    cdef cnp.float64_t t[3]
    t[0] = (voxel_size[0] / 2) / direction[0]
    t[1] = (voxel_size[1] / 2) / direction[1]
    t[2] = (voxel_size[2] / 2) / direction[2]

    cdef cnp.float64_t delta_t[3]
    delta_t[0] = voxel_size[0] / direction[0]
    delta_t[1] = voxel_size[1] / direction[1]
    delta_t[2] = voxel_size[2] / direction[2]

    cdef cnp.int32_t xmax = grid_shape[0]
    cdef cnp.int32_t ymax = grid_shape[1]
    cdef cnp.int32_t zmax = grid_shape[2]

    # TODO Consider the size of these array better
    voxels_traversed = np.zeros((xmax + ymax + zmax, 3), dtype=np.int32)
    cdef cnp.int32_t [:, :] voxels_traversed_mv = voxels_traversed
    intersection_t_values = np.zeros(xmax + ymax + zmax + 10, dtype=np.float64)
    cdef cnp.float64_t [:] intersection_t_values_mv = intersection_t_values
    
    cdef cnp.int32_t v_count = 0 
    cdef cnp.int32_t i_count = 0
    
    while (current_voxel[0] >= 0 and current_voxel[0] < xmax and
           current_voxel[1] >= 0 and current_voxel[1] < ymax and
           current_voxel[2] >= 0 and current_voxel[2] < zmax):

        voxels_traversed[v_count][0] = current_voxel[0]
        voxels_traversed[v_count][1] = current_voxel[1]
        voxels_traversed[v_count][2] = current_voxel[2]
        if t[0] < t[1]:
            if t[0] < t[2]:
                intersection_t_values[i_count] = t[0]
                t[0] += delta_t[0]
                current_voxel[0] = current_voxel[0] + step[0]
            else:
                intersection_t_values[i_count] = t[2]
                t[2] += delta_t[2]
                current_voxel[2] = current_voxel[2] + step[2]
        else:
            if t[1] < t[2]:
                intersection_t_values[i_count] = t[1]
                t[1] += delta_t[1]
                current_voxel[1] = current_voxel[1] + step[1]
            else:
                intersection_t_values[i_count] = t[2]
                t[2] += delta_t[2]
                current_voxel[2] = current_voxel[2] + step[2]
        v_count = v_count + 1
        i_count = i_count + 1

    return (np.asarray(intersection_t_values_mv[:i_count]), np.asarray(voxels_traversed_mv[:v_count, :]))
