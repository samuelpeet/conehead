cmake_minimum_required(VERSION 3.15)
project(conehead_gpu LANGUAGES CXX CUDA)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75 86)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

pybind11_add_module(
    conehead_gpu SHARED
    src/gpu/oad.cu
    src/gpu/d_geo.cu
    src/gpu/d_eff.cu
    src/gpu/fluence.cu
    src/gpu/terma.cu
    src/gpu/mask.cu
    src/gpu/dose.cu
    src/gpu/bindings.cu
)

target_link_libraries(conehead_gpu PRIVATE pybind11::module CUDA::cudart)

set_target_properties(conehead_gpu PROPERTIES PREFIX "")