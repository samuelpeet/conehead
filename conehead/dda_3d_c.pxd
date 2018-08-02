cimport numpy as cnp
from .vector cimport vector

cdef struct result:
    vector intersection_t_values
    vector voxels_traversed

cdef void dda_3d_c(cnp.float64_t* direction, cnp.int32_t* grid_shape, cnp.int32_t* first_voxel, cnp.float64_t* voxel_size, result* r) nogil