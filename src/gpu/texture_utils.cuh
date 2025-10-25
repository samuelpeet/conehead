#ifndef CONEHEAD_GPU_TEXTURE_UTILS_H
#define CONEHEAD_GPU_TEXTURE_UTILS_H

#include <cstring>
#include <cuda_runtime.h>

// Small RAII-like struct to hold a texture object and its backing CUDA array.
struct Texture3DHandle {
    cudaTextureObject_t tex = 0;
    cudaArray_t array = nullptr;
};

/**
 * Create a 3D texture object from a host pointer containing contiguous
 * float voxels in (nx,ny,nz) layout (x fastest). Returns a Texture3DHandle
 * with texture object and underlying cudaArray. The caller must call
 * destroy_texture3d() to free resources.
 *
 * Note: This helper uses non-normalized coordinates (tex.normalizedCoords=0)
 * so sampling should pass texel coordinates (0..nx-1 etc.).
 */
static inline Texture3DHandle create_texture3d_from_ptr(const float* host_ptr, int nx, int ny, int nz,
    cudaTextureFilterMode filter = cudaFilterModeLinear)
{
    Texture3DHandle h {};

    cudaExtent extent = make_cudaExtent((size_t)nx, (size_t)ny, (size_t)nz);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray = nullptr;
    cudaMalloc3DArray(&cuArray, &channelDesc, extent);

    // Copy host memory into 3D array. Host is tightly packed (x fastest).
    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr((void*)host_ptr, nx * sizeof(float), nx, ny);
    copyParams.dstArray = cuArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    // Resource and texture descriptors
    cudaResourceDesc resDesc;
    std::memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    std::memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = filter;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0; // use unnormalized coordinates

    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);

    h.tex = tex;
    h.array = cuArray;
    return h;
}

static inline void destroy_texture3d(Texture3DHandle& h)
{
    if (h.tex)
        cudaDestroyTextureObject(h.tex);
    if (h.array)
        cudaFreeArray(h.array);
    h.tex = 0;
    h.array = nullptr;
}

// Small RAII-like struct to hold a 2D texture object and its backing CUDA array.
struct Texture2DHandle {
    cudaTextureObject_t tex = 0;
    cudaArray_t array = nullptr;
};

/**
 * Create a 2D texture object from a host pointer containing contiguous
 * float pixels in (nx,ny) layout (x fastest). Returns a Texture2DHandle
 * with texture object and underlying cudaArray. The caller must call
 * destroy_texture2d() to free resources.
 *
 * Note: This helper uses non-normalized coordinates (tex.normalizedCoords=0)
 * so sampling should pass texel coordinates (0..nx-1 etc.).
 */
static inline Texture2DHandle create_texture2d_from_ptr(const float* host_ptr, int nx, int ny,
    cudaTextureFilterMode filter = cudaFilterModeLinear)
{
    Texture2DHandle h {};

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray = nullptr;
    cudaMallocArray(&cuArray, &channelDesc, nx, ny);

    // Copy host memory into 2D array. Host is tightly packed (x fastest).
    size_t spitch = nx * sizeof(float);
    cudaMemcpy2DToArray(cuArray, 0, 0, host_ptr, spitch, spitch, ny, cudaMemcpyHostToDevice);

    // Resource and texture descriptors
    cudaResourceDesc resDesc;
    std::memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    std::memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = filter;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0; // use unnormalized coordinates

    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);

    h.tex = tex;
    h.array = cuArray;
    return h;
}

static inline void destroy_texture2d(Texture2DHandle& h)
{
    if (h.tex)
        cudaDestroyTextureObject(h.tex);
    if (h.array)
        cudaFreeArray(h.array);
    h.tex = 0;
    h.array = nullptr;
}

#endif // CONEHEAD_GPU_TEXTURE_UTILS_H
