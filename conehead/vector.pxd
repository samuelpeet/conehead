cimport numpy as cnp
from libc.stdlib cimport malloc, realloc, free

cdef struct vector:
    cnp.int32_t size
    cnp.int32_t capacity
    cnp.float64_t* data

cdef vector_init(vector* vector)

cdef vector_append(vector* vector, cnp.int32_t value)

cdef vector_get(vector* vector, cnp.int32_t index)

cdef vector_set(vector* vector, cnp.int32_t index, cnp.float64_t value)

cdef vector_double_capacity_if_full(vector* vector)

cdef vector_free(vector* vector)