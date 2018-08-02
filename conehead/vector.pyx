
cdef void vector_init(vector* v) nogil:
    # Initialize size and capacity
    v.size = 0
    v.capacity = 10

    # Allocate memory for v.data
    v.data = <void**>malloc(sizeof(void*) * v.capacity)


cdef void vector_append(vector* v, void* value) nogil:
    # Make sure there's room to expand into
    vector_double_capacity_if_full(v)

    # Append the value and increment vector.size
    v.data[v.size] = value
    v.size += 1


cdef void* vector_get(vector* v, cnp.int32_t index) nogil:
    if index >= v.size or index < 0:
        printf("Index %d out of bounds for vector of size %d\n", index, v.size)
        exit(1)
    return v.data[index]


cdef void vector_set(vector* v, cnp.int32_t index, void* value) nogil:
    # Set the value at the desired index
    v.data[index] = value

cdef cnp.int32_t vector_size(vector* v) nogil:
    # Return size of vector
    return v.size

cdef void vector_double_capacity_if_full(vector* v) nogil:

    cdef void** data

    if (v.size >= v.capacity):
        # double v.capacity and resize the allocated memory accordingly
        v.capacity *= 2
        data = <void**>realloc(v.data, sizeof(void *) * v.capacity)
        if data:
            v.data = data
        else:
            printf("Error reallocating vector memory")
            exit(1)


cdef void vector_free(vector* v) nogil:
    cdef cnp.int32_t i
    for i in range(v.size):
        free(v.data[i])
    free(v.data)
