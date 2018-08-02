cimport numpy as cnp
from libc.stdlib cimport malloc, realloc, free, exit
from libc.stdio cimport sprintf, puts, printf

cdef struct vector:
    cnp.int32_t size
    cnp.int32_t capacity
    void** data

cdef void vector_init(vector* v) nogil

cdef void vector_append(vector* v, void* value) nogil

cdef void* vector_get(vector* v, cnp.int32_t index) nogil

cdef void vector_set(vector* v, cnp.int32_t index, void* value) nogil

cdef cnp.int32_t vector_size(vector* v) nogil

cdef void vector_double_capacity_if_full(vector* v) nogil

cdef void vector_free(vector* v) nogil