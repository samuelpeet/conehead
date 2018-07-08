import numpy as np
import cython
cimport numpy as cnp

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def dda_3d_c(cnp.float64_t[:] source, cnp.float64_t[:] direction, cnp.float64_t[:,:,:] grid, cnp.int32_t[:] current_voxel, cnp.float64_t[:] voxel_size):

    cdef cnp.int32_t step[3]
    step[0] = -1 if direction[0] < 0 else 1
    step[1] = -1 if direction[1] < 0 else 1
    step[2] = -1 if direction[2] < 0 else 1
    
    cdef cnp.int32_t vox[3]
    vox[0] = current_voxel[0]
    vox[1] = current_voxel[1]
    vox[2] = current_voxel[2]

    cdef cnp.float64_t t[3]
    t[0] = (voxel_size[0] / 2) / direction[0]
    t[1] = (voxel_size[1] / 2) / direction[1]
    t[2] = (voxel_size[2] / 2) / direction[2]

    cdef cnp.float64_t delta_t[3]
    delta_t[0] = voxel_size[0] / direction[0]
    delta_t[1] = voxel_size[1] / direction[1]
    delta_t[2] = voxel_size[2] / direction[2]

    cdef cnp.float64_t xmax = grid.shape[0]
    cdef cnp.float64_t ymax = grid.shape[1]
    cdef cnp.float64_t zmax = grid.shape[2]

    # TODO fix these magic numbers
    cdef cnp.int32_t voxels_traversed[200][3]
    cdef cnp.float64_t intersection_t_values[200]
    
    cdef cnp.int32_t v_count = 0 
    cdef cnp.int32_t i_count = 0
    
    while (vox[0] >= 0 and vox[0] < xmax and
           vox[1] >= 0 and vox[1] < ymax and
           vox[2] >= 0 and vox[2] < zmax):

        voxels_traversed[v_count][0] = vox[0]
        voxels_traversed[v_count][1] = vox[1]
        voxels_traversed[v_count][2] = vox[2]
        if t[0] < t[1]:
            if t[0] < t[2]:
                intersection_t_values[i_count] = t[0]
                t[0] += delta_t[0]
                vox[0] = vox[0] + step[0]
            else:
                intersection_t_values[i_count] = t[2]
                t[2] += delta_t[2]
                vox[2] = vox[2] + step[2]
        else:
            if t[1] < t[2]:
                intersection_t_values[i_count] = t[1]
                t[1] += delta_t[1]
                vox[1] = vox[1] + step[1]
            else:
                intersection_t_values[i_count] = t[2]
                t[2] += delta_t[2]
                vox[2] = vox[2] + step[2]
        v_count = v_count + 1
        i_count = i_count + 1
    
    return intersection_t_values
