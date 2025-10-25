// Include headers (do not include .cu files here)
#include "conehead.cuh"
#include "device_functions.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace py = pybind11;

PYBIND11_MODULE(conehead_gpu, m)
{
    m.def("oad", &map_oad, py::arg("oad_grid"), py::arg("num_voxels"),
        py::arg("corner"), py::arg("resolution"), py::arg("source_position"),
        py::arg("source_v_x"), py::arg("source_v_y"), py::arg("source_v_z"));
    m.def("d_geo", &map_d_geo, py::arg("d_geo_grid"), py::arg("num_voxels"),
        py::arg("corner"), py::arg("resolution"), py::arg("source_position"));
    m.def("d_eff", &map_d_eff, py::arg("d_eff_grid"), py::arg("num_voxels"),
        py::arg("corner"), py::arg("resolution"), py::arg("density_grid"), py::arg("source_position"));
    m.def("fluence", &map_fluence, py::arg("fluence_grid"), py::arg("fluence_map_pri"),
        py::arg("fluence_map_sec"), py::arg("num_voxels"), py::arg("corner"), py::arg("resolution"),
        py::arg("d_geo_grid"), py::arg("source_position"), py::arg("source_v_x"), py::arg("source_v_y"),
        py::arg("source_v_z"), py::arg("source_sad"), py::arg("pri_s"), py::arg("pri_x"), py::arg("pri_y"),
        py::arg("pri_z"), py::arg("sec_s"), py::arg("sec_x"), py::arg("sec_y"), py::arg("sec_z"),
        py::arg("samples"));
    m.def("terma", &map_terma, py::arg("terma_grid"), py::arg("fluence_grid"), py::arg("d_geo_grid"),
        py::arg("d_eff_grid"), py::arg("num_voxels"), py::arg("num_energies"), py::arg("energies"),
        py::arg("energy_weights"), py::arg("mu_w"), py::arg("source_sad"),
        py::arg("oad_grid"), py::arg("off_axis_softening_fs_interp"), py::arg("off_axis_softening_dx"));
    m.def("mask", &map_mask, py::arg("mask_grid"), py::arg("terma_grid"), py::arg("num_voxels"),
        py::arg("resolution"), py::arg("max_distance_cm"), py::arg("terma_threshold"));
    m.def("dose", &map_dose, py::arg("dose_grid"), py::arg("resolution"), py::arg("num_voxels"), py::arg("corner"),
        py::arg("density_grid"), py::arg("d_geo_grid"), py::arg("terma_grid"),
        py::arg("mask_grid"), py::arg("kernel_thetas"), py::arg("kernel_phis"),
        py::arg("kernel_omegas"), py::arg("kernel"), py::arg("source_sad"), py::arg("source_position"),
        py::arg("source_v_x"), py::arg("source_v_y"), py::arg("source_v_z"),
        py::arg("n_depth_bins"), py::arg("kernel_depth_res_cm"), py::arg("max_kernel_depth_cm"), py::arg("ds_cm"));
}