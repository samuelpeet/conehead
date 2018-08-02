import cython
cimport numpy as cnp
from conehead.vector cimport  (
    vector, vector_init, vector_append, vector_get, vector_set, vector_free,
    vector_size
)
from libc.stdio cimport sprintf, puts, printf
from libc.stdlib cimport malloc

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)  # Deactivate negative indexing
@cython.cdivision(True)
cdef void dda_3d_c(cnp.float64_t* direction, cnp.int32_t* grid_shape,
                   cnp.int32_t* first_voxel, cnp.float64_t* voxel_size,
                   result* r) nogil:
    """ Calculate the intersection points of a ray with a voxel grid, using a
    3D DDA algorithm. See Amanatides & Woo (1987) Eurographics 87(3).

    Parameters
    ----------
    direction : cnp.float64_t*
        Direction vector of the ray, in form [dx, dy, dz]
    grid_shape : cnp.int32_t*
        Shape of 3D voxel grid, in form [len(x), len(y), len(z)]
    first_voxel : cnp.int32_t*
        Index of ray source voxel, in form [x, y, z]
    voxel_size : cnp.float64_t*
        Size of voxel dimensions, in form [dim_x, dim_y, dim_z]
    r : result*
        Pointer to a result struct for storing the array of traversed voxels
        and the array of intersection t-values
    """
    cdef cnp.int32_t step[3]
    step[0] = -1 if direction[0] < 0 else 1
    step[1] = -1 if direction[1] < 0 else 1
    step[2] = -1 if direction[2] < 0 else 1

    if direction[0] < 0:
        direction[0] = -1 * direction[0]
    if direction[1] < 0:
        direction[1] = -1 * direction[1]
    if direction[2] < 0:
        direction[2] = -1 * direction[2]

    cdef cnp.int32_t current_voxel[3]
    current_voxel[0] = first_voxel[0]
    current_voxel[1] = first_voxel[1]
    current_voxel[2] = first_voxel[2]

    cdef cnp.float64_t t[3]
    cdef cnp.float64_t delta_t[3]
    cdef cnp.float64_t big_number = 1000000000
    if direction[0] == 0.0:
        t[0] = big_number
        delta_t[0] = big_number
    else:
        t[0] = (voxel_size[0] / 2) / direction[0]
        delta_t[0] = voxel_size[0] / direction[0]
    if direction[1] == 0.0:
        t[1] = big_number
        delta_t[1] = big_number
    else:
        t[1] = (voxel_size[1] / 2) / direction[1]
        delta_t[1] = voxel_size[1] / direction[1]
    if direction[2] == 0.0:
        t[2] = big_number
        delta_t[2] = big_number
    else:
        t[2] = (voxel_size[2] / 2) / direction[2]
        delta_t[2] = voxel_size[2] / direction[2]

    cdef cnp.int32_t xmax = grid_shape[0]
    cdef cnp.int32_t ymax = grid_shape[1]
    cdef cnp.int32_t zmax = grid_shape[2]

    while (current_voxel[0] >= 0 and current_voxel[0] < xmax and
           current_voxel[1] >= 0 and current_voxel[1] < ymax and
           current_voxel[2] >= 0 and current_voxel[2] < zmax):

        vector_append(&r.voxels_traversed, copy_v(current_voxel))

        if t[0] < t[1]:
            if t[0] < t[2]:
                vector_append(&r.intersection_t_values, copy_t(&t[0]))
                t[0] += delta_t[0]
                current_voxel[0] = current_voxel[0] + step[0]
            else:
                vector_append(&r.intersection_t_values, copy_t(&t[2]))
                t[2] += delta_t[2]
                current_voxel[2] = current_voxel[2] + step[2]
        else:
            if t[1] < t[2]:
                vector_append(&r.intersection_t_values, copy_t(&t[1]))
                t[1] += delta_t[1]
                current_voxel[1] = current_voxel[1] + step[1]
            else:
                vector_append(&r.intersection_t_values, copy_t(&t[2]))
                t[2] += delta_t[2]
                current_voxel[2] = current_voxel[2] + step[2]


cdef cnp.float64_t* copy_t(cnp.float64_t* t) nogil:
    """ Copy a t-value.

    Parameters
    ----------
    t : cnp.float64_t*
        Pointer to the value to copy

    Returns
    -------
    cnp.float64_t*
        Pointer to the newly created copy
    """
    cdef cnp.float64_t* new_t = <cnp.float64_t*>malloc(sizeof(cnp.float64_t))
    new_t[0] = t[0]
    return new_t


cdef cnp.int32_t* copy_v(cnp.int32_t* v) nogil:
    """ Copy a voxel index position array.

    Parameters
    ----------
    v : cnp.int32_t*
        Pointer to the voxel to copy

    Returns
    -------
    cnp.int32_t*
        Pointer to the newly created array copy
    """
    cdef cnp.int32_t* new_v = <cnp.int32_t*>malloc(3 * sizeof(cnp.int32_t))
    new_v[0] = v[0]
    new_v[1] = v[1]
    new_v[2] = v[2]
    return new_v
