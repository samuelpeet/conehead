import numpy as np
cimport cython
cimport numpy as cnp
from libc.math cimport sin, cos, ceil, floor
from libc.stdio cimport sprintf, printf, puts, fflush, stdout
from libc.stdlib cimport malloc
from conehead.dda_3d_c cimport dda_3d_c, result
from conehead.vector cimport (
    vector, vector_init, vector_append, vector_get, vector_set, vector_free,
    vector_size
)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def convolve_c(cnp.float64_t[:,:,:] dose_grid_terma,
               cnp.float64_t[:,:,:] dose_grid_dose,
               cnp.float64_t[:] dose_grid_dim, cnp.float64_t[:] thetas,
               cnp.float64_t[:] phis, kernel):
    """ Calculate 3D grid of doses by convolving a cumulative energy deposition
    kernel with a 3D grid of TERMA values.

    Parameters
    ----------
    dose_grid_terma : ndarray (memoryview)
        Grid of Terma values (n_dim = 3)
    dose_grid_dose : ndarray (memoryview)
        Grid of dose values (n_dim = 3). Calculated doses are added directly
        to this array.
    dose_grid_dim : ndarray (memoryview)
        Voxel dimensions, in form [dim_x, dim_y, dim_z]
    thetas : ndarray (memoryview)
        Azimuthal kernel cone angles (n_dim = 1)
    phis : ndarray (memoryview)
        Altitudinal kernel cone angles (n_dim = 1)
    kernel : dict
        Dictionary holding cumulative kernel data
    """
    cdef cnp.int32_t xlen = dose_grid_terma.shape[0]
    cdef cnp.int32_t ylen = dose_grid_terma.shape[1]
    cdef cnp.int32_t zlen = dose_grid_terma.shape[2]
    cdef cnp.int32_t dose_grid_shape[3]
    dose_grid_shape = [
        xlen, ylen, zlen
    ]
    cdef cnp.float64_t T, N, N_inv, theta_rad, phi_rad, c_t, s_t, c_p, s_p
    cdef cnp.float64_t pi = 3.14159265359
    cdef cnp.int32_t num_thetas = thetas.shape[0]
    cdef cnp.int32_t num_phis = phis.shape[0]
    cdef cnp.int32_t x, y, z, i, j, m
    cdef cnp.int32_t current_voxel[3]
    cdef cnp.float64_t direction[3]
    cdef cnp.float64_t dimensions[3]
    dimensions = [
        dose_grid_dim[0],
        dose_grid_dim[1],
        dose_grid_dim[2]
    ]
    cdef result r
    cdef result* r_ptr = &r
    cdef cnp.float64_t intersection
    cdef cnp.int32_t intersection_index
    cdef vector intersection_indices
    cdef char key[10]
    cdef vector kernel_data
    vector_init(&kernel_data)
    process_kernel(kernel, phis, &kernel_data)
    cdef vector* current_cone_kernel_data
    cdef cnp.int32_t* v
    cdef cnp.float64_t* k1
    cdef cnp.float64_t* k2
    cdef cnp.float64_t k3
    cdef cnp.int32_t index

    # Iterate through all voxels
    for x in range(xlen):
        printf("%d/%d...", x, xlen)  # Naive progress indicator
        fflush(stdout)
        for y in range(ylen):
            for z in range(zlen):

                # Only convolve if the current voxel has non-zero TERMA
                T = dose_grid_terma[x, y, z]
                if T:

                    # Iterate through each cone
                    for i in range(num_thetas):
                        for j in range(num_phis):

                            # Save current voxel index for later
                            current_voxel = [x, y, z]

                            # Calculate direction vector
                            theta_rad = thetas[i] * pi / 180.0
                            phi_rad = phis[j] * pi / 180.0
                            c_t = cos(theta_rad)
                            s_t = sin(theta_rad)
                            c_p = cos(phi_rad)
                            s_p = sin(phi_rad)
                            direction = [c_t * s_p, s_t * s_p, c_p]
                            N = (
                                direction[0] * direction[0] +
                                direction[1] * direction[1] +
                                direction[2] * direction[2]
                            )
                            N_inv = 1.0 / N
                            direction = [  # Normalise
                                direction[0] * N_inv,
                                direction[1] * N_inv,
                                direction[2] * N_inv
                            ]
                            direction = [  # Discretise
                                ceil(direction[0] * 100000) * 0.00001,
                                ceil(direction[1] * 100000) * 0.00001,
                                ceil(direction[2] * 100000) * 0.00001
                            ]

                            # Perform raytracing to find voxels along cone line
                            # and boundary intersection values
                            vector_init(&r_ptr.intersection_t_values)
                            vector_init(&r_ptr.voxels_traversed)
                            dda_3d_c(
                                direction,
                                dose_grid_shape,
                                current_voxel,
                                dimensions,
                                &r
                            )

                            # Calculate which indices in kernel data array will
                            # correspond to boundary intersection values
                            vector_init(&intersection_indices)
                            for m in range(r.intersection_t_values.size):
                                intersection = (<cnp.float64_t*>vector_get(
                                    &r.intersection_t_values, m
                                ))[0]
                                intersection_index = <cnp.int32_t>floor(
                                    intersection * 100.0 - 50.0
                                )
                                intersection_index = abs(intersection_index)
                                vector_append(
                                    &intersection_indices,
                                    copy_i(&intersection_index)
                                )

                            # Find corresponding kernel data value for each
                            # voxel intersected by cone line
                            current_cone_kernel_data = <vector*>vector_get(
                                &kernel_data, j
                            )
                            for m in range(intersection_indices.size):
                                v = <cnp.int32_t*>vector_get(
                                    &r.voxels_traversed, m
                                )
                                if m == 0:
                                    # First voxel, get appropriate kernal index
                                    # of and corresponding kernel value
                                    # (single intersection)
                                    index1 = (<cnp.int32_t*>vector_get(
                                        &intersection_indices, m
                                    ))[0]
                                    k3 = (<cnp.float64_t*>vector_get(
                                        current_cone_kernel_data, index1
                                    ))[0]
                                else:
                                    # For al other voxels, get the diffeence
                                    # between cumulative kernal data across the
                                    # traverse of the voxel
                                    # (double intersection)
                                    index1 = (<cnp.int32_t*>vector_get(
                                        &intersection_indices, m
                                    ))[0]
                                    k1 = <cnp.float64_t*>vector_get(
                                        current_cone_kernel_data, index1
                                    )
                                    index2 = (<cnp.int32_t*>vector_get(
                                        &intersection_indices, m - 1
                                    ))[0]
                                    k2 = <cnp.float64_t*>vector_get(
                                        current_cone_kernel_data, index2
                                    )
                                    k3 = k1[0] - k2[0]

                                dose_grid_dose[v[0], v[1], v[2]] += T * k3

                            # Free memory in all arrays
                            vector_free(&r.intersection_t_values)
                            vector_free(&r.voxels_traversed)
                            vector_free(&intersection_indices)


cdef cnp.int32_t* copy_i(cnp.int32_t* i) nogil:
    """ Copy an index.

    Parameters
    ----------
    i : cnp.int32_t*
        Pointer to the index to copy

    Returns
    -------
    cnp.int32_t*
        Pointer to the newly created copy
    """
    cdef cnp.int32_t* new_i = <cnp.int32_t*>malloc(sizeof(cnp.int32_t))
    new_i[0] = i[0]
    return new_i


cdef cnp.float64_t* copy_f(cnp.float64_t* f) nogil:
    """ Copy a float.

    Parameters
    ----------
    f : cnp.float64_t*
        Pointer to the float to copy

    Returns
    -------
    cnp.float64_t*
        Pointer to the newly created copy
    """
    cdef cnp.float64_t* new_f = <cnp.float64_t*>malloc(sizeof(cnp.float64_t))
    new_f[0] = f[0]
    return new_f


cdef void process_kernel(kernel, phis, vector* kernel_data):
    """ Convert kernel data from python dictionary to vector.

    Parameters
    ----------
    kernel : dict
        Dictionary of kernel data
    phis : ndarray (memoryview)
        Altitudinal kernel cone angles (n_dim = 1)
    kernel_data : vector*
        Pointer to vector that will hold the converted data
    """
    cdef vector* single_cone_data

    for phi in phis:
        single_cone_data = <vector*>malloc(sizeof(vector))
        vector_init(single_cone_data)
        get_single_cone_data(kernel, phi, single_cone_data)
        vector_append(kernel_data, single_cone_data)


cdef void get_single_cone_data(kernel, phi, vector* v):
    """ Fill a vector with kernel data from a single cone line..

    Parameters
    ----------
    kernel : dict
        Dictionary of kernel data
    phi : cnp.float64_t
        Altitudinal angle of cone
    v : vector*
        Pointer to vector that will hold the converted data
    """
    cdef cnp.int32_t N = len(kernel.cumulative["{:.2f}".format(phi)])
    cdef cnp.float64_t value

    for n in range(N):
        value = kernel.cumulative["{:.2f}".format(phi)][n]
        vector_append(v, copy_f(&value))
