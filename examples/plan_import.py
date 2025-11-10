# %%
import pydicom
import numpy as np
import toml
from conehead.exam import Exam
from conehead.plan import Plan
from conehead.grid import Grid
from conehead.block import Block
from conehead.source import Source
from conehead.kernel import Kernel
from conehead.nist import mu_water
import conehead_gpu as gpu


# Load machine settings
settings = toml.load("Truebeam_6FFF_M120.toml")

# Load CT and structure set
dicom_dir = "Prostate 3DCRT/CT"
exam = Exam(dicom_folder=f"{dicom_dir}", hu_lut_path="Siemens_Confidence.toml")

# Load RT Plan
dicom_path = f"{dicom_dir}/RP1.2.752.243.1.1.20251030132425394.9900.34053.dcm"
plan = Plan(dicom_path)

# Define grid geometry for dose calculation
grid = Grid(
    corner=np.array([-26.96, -23.12, -10.20], dtype=np.float32),
    resolution=np.array([0.2, 0.2, 0.2], dtype=np.float32),
    num_voxels=np.array([267, 206, 138], dtype=np.int32),
)

if plan.delivery_method == "SMLC":
    for beam in plan.beams:
        if beam.type != "STATIC":
            continue  # Only process static beams for now

        # Initialise the source geometry
        source = Source()
        source.gantry = beam.control_points[0].gantry
        source.collimator = beam.control_points[0].collimator

        # Initalise the block and compute the fluence maps
        block = Block(control_point=beam.control_points[0], settings=settings)
        fluence_map_pri, fluence_map_sec = block.get_fluence_maps()

        # Initialise the dose calculation kernel
        kernel = Kernel(settings=settings)

        # Allocate GPU buffers
        oad_grid = np.zeros(grid.num_voxels, dtype=np.float32)
        d_geo_grid = np.zeros(grid.num_voxels, dtype=np.float32)
        d_eff_grid = np.zeros(grid.num_voxels, dtype=np.float32)
        density_grid = exam.densities_on_grid(grid).values
        fluence_grid = np.zeros(grid.num_voxels, dtype=np.float32)
        terma_grid = np.zeros(grid.num_voxels, dtype=np.float32)
        mask_grid = np.zeros(grid.num_voxels, dtype=np.float32)
        dose_grid = np.zeros(grid.num_voxels, dtype=np.float32)

        # Perform GPU calculations
        gpu.oad(
            oad_grid=oad_grid,
            num_voxels=grid.num_voxels,
            corner=grid.corner,
            resolution=grid.resolution,
            source_position=source.position,
            source_v_x=source.v_x,
            source_v_y=source.v_y,
            source_v_z=source.v_z,
        )
        gpu.d_geo(
            d_geo_grid=d_geo_grid,
            num_voxels=grid.num_voxels,
            corner=grid.corner,
            resolution=grid.resolution,
            source_position=source.position,
        )
        gpu.d_eff(
            d_eff_grid=d_eff_grid,
            num_voxels=grid.num_voxels,
            corner=grid.corner,
            resolution=grid.resolution,
            density_grid=density_grid,
            source_position=source.position,
        )
        gpu.fluence(
            fluence_grid=fluence_grid,
            fluence_map_pri=fluence_map_pri,
            fluence_map_sec=fluence_map_sec,
            num_voxels=grid.num_voxels,
            corner=grid.corner,
            resolution=grid.resolution,
            d_geo_grid=d_geo_grid,
            source_position=source.position,
            source_v_x=source.v_x,
            source_v_y=source.v_y,
            source_v_z=source.v_z,
            source_sad=source.sad,
            pri_z=np.float32(settings["sources"]["pri_z"]),
            sec_z=np.float32(settings["sources"]["sec_z"]),
            samples=np.int32(settings["sources"]["samples"]),
        )
        gpu.terma(
            terma_grid=terma_grid,
            fluence_grid=fluence_grid,
            d_geo_grid=d_geo_grid,
            d_eff_grid=d_eff_grid,
            num_voxels=grid.num_voxels,
            num_energies=np.int32(len(settings["energy_spectrum"]["energies"])),
            energies=np.array(settings["energy_spectrum"]["energies"], dtype=np.float32),
            energy_weights=np.array(settings["energy_spectrum"]["weights"], dtype=np.float32),
            mu_w=mu_water(np.array(settings["energy_spectrum"]["energies"], dtype=np.float32)),
            source_sad=source.sad,
            oad_grid=oad_grid,
            off_axis_softening_fs_interp=np.array(
                settings["off_axis_softening"]["factors"], dtype=np.float32
            ),
            off_axis_softening_dx=np.float32(40.0),
        )
        if settings["calculation"]["terma_mask_enable"]:
            gpu.mask(
                mask_grid=mask_grid,
                terma_grid=terma_grid,
                num_voxels=grid.num_voxels,
                resolution=grid.resolution,
                max_distance_cm=np.float32(settings["calculation"]["terma_mask_max_distance"]),
                terma_threshold=np.float32(
                    settings["calculation"]["terma_mask_terma_threshold"] * terma_grid.max()
                ),
            )
        else:
            mask_grid.fill(1.0)

        gpu.dose(
            dose_grid=dose_grid,
            resolution=grid.resolution,
            num_voxels=grid.num_voxels,
            corner=grid.corner,
            density_grid=density_grid,
            d_geo_grid=d_geo_grid,
            terma_grid=terma_grid,
            mask_grid=mask_grid,
            kernel_thetas=kernel.thetas,
            kernel_phis=kernel.phis,
            kernel_omegas=kernel.omegas,
            kernel=kernel.values,
            source_sad=source.sad,
            source_position=source.position,
            source_v_x=source.v_x,
            source_v_y=source.v_y,
            source_v_z=source.v_z,
            n_depth_bins=kernel.n_depth_bins,
            kernel_depth_res_cm=kernel.kernel_depth_res_cm,
            max_kernel_depth_cm=kernel.max_kernel_depth_cm,
            ds_cm=kernel.ds_cm,
        )

        # Apply output factor correction and normalisation
        x_gap = np.abs(
            beam.control_points[0].jaw_x_positions[1] - beam.control_points[0].jaw_x_positions[0]
        )
        y_gap = np.abs(
            beam.control_points[0].jaw_y_positions[1] - beam.control_points[0].jaw_y_positions[0]
        )
        equiv_square = 2 * x_gap * y_gap / (x_gap + y_gap)
        ofc = np.interp(
            equiv_square,
            settings["output_factor_correction"]["field_sizes"],
            settings["output_factor_correction"]["factors"],
        ).astype(np.float32)
        N = np.float32(settings["calculation"]["normalisation"])
        MU = beam.mu
        beam.dose_values = dose_grid * ofc * N * MU

# %%
