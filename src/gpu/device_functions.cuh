#ifndef CONEHEAD_GPU_DEVICE_FUNCTIONS_CUH
#define CONEHEAD_GPU_DEVICE_FUNCTIONS_CUH

#include <cuda_runtime.h>

// Small, header-only device helpers used by multiple CUDA translation units.
// Define as static inline / __device__ so they can be included into any .cu
// file without requiring separable device compilation (-rdc) or a device-link
// step.

/**
 * @brief Compute dot product of two float3 vectors.
 */
static __device__ __forceinline__ float dot3_device(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * @brief Ray-plane intersection helper on device.
 *
 * Writes the intersection point into out_pos_plane and returns true on a
 * successful intersection. If the ray is parallel (or nearly parallel) to
 * the plane the function returns false and out_pos_plane is left untouched.
 */
static __device__ __forceinline__ bool line_plane_collision_device(float3* out_pos_plane,
    const float3& ray_start,
    const float3& ray_direction,
    const float3& plane_normal,
    const float3& plane_point,
    float epsilon)
{
    float ndotu = dot3_device(plane_normal, ray_direction);
    if (fabsf(ndotu) < epsilon) {
        return false;
    }
    // Plane equation: dot(normal, point - plane_point) = 0
    float3 ray_to_plane;
    ray_to_plane.x = plane_point.x - ray_start.x;
    ray_to_plane.y = plane_point.y - ray_start.y;
    ray_to_plane.z = plane_point.z - ray_start.z;
    float si = dot3_device(plane_normal, ray_to_plane) / ndotu;
    out_pos_plane->x = ray_start.x + si * ray_direction.x;
    out_pos_plane->y = ray_start.y + si * ray_direction.y;
    out_pos_plane->z = ray_start.z + si * ray_direction.z;
    return true;
}

/**
 * @brief Lookup in a flattened fluence map (header-only device helper).
 *
 * Assumptions: map is width x height (defaults in code were 560), input
 * position is in cm and map index mapping may apply a fixed offset. Keep
 * this small helper local to avoid repeating mapping logic across TUs.
 */
static __device__ __forceinline__ float fluence_map_lookup_device(float2 position, cudaTextureObject_t fluence_tex)
{
    // Convert world coords (cm) to texture coordinates (mm index space).
    // Mapping: mm_index = position_cm*10 + 280, sample at texel center (index + 0.5).
    float ix = position.x * 10.0f + 280.0f; // mm index along X
    float iy = position.y * 10.0f + 280.0f; // mm index along Y
    if (ix < 0.0f || ix > 559.0f || iy < 0.0f || iy > 559.0f) {
        return 0.0f;
    }
    // Sample the 2D texture at the texel center.
    return tex2D<float>(fluence_tex, ix + 0.5f, iy + 0.5f);
}

#endif // CONEHEAD_GPU_DEVICE_FUNCTIONS_CUH
