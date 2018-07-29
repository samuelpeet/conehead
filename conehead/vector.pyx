cdef vector_init(vector* vector):
    # Initialize size and capacity
    vector.size = 0
    vector.capacity = 10

    # Allocate memory for vector.data
    vector.data = <cnp.float64_t*>malloc(sizeof(cnp.float64_t) * vector.capacity)


cdef vector_append(vector* vector, cnp.int32_t value):
    # Make sure there's room to expand into
    vector_double_capacity_if_full(vector)

    # Append the value and increment vector.size
    vector.data[vector.size] = value
    vector.size += 1


cdef vector_get(vector* vector, cnp.int32_t index):
#   if (index >= vector.size || index < 0) {
#     printf("Index %d out of bounds for vector of size %d\n", index, vector.size)
#     exit(1)
#   }
    return vector.data[index]


cdef vector_set(vector* vector, cnp.int32_t index, cnp.float64_t value):
    # Set the value at the desired index
    vector.data[index] = value


cdef vector_double_capacity_if_full(vector* vector):
    if (vector.size >= vector.capacity):
        # double vector.capacity and resize the allocated memory accordingly
        vector.capacity *= 2
        vector.data = <cnp.float64_t*>realloc(vector.data, sizeof(cnp.float64_t) * vector.capacity)


cdef vector_free(vector* vector):
    free(vector.data)
