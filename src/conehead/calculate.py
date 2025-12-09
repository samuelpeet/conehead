import numpy as np
import numpy.typing as npt
from conehead.kernel import Kernel
from conehead.nist import mu_water
from conehead.grid import Grid
from conehead.exam import Exam
from conehead.source import Source
import conehead_gpu as gpu

import matplotlib.pyplot as plt

def calculate(
    grid: Grid,
    source: Source,
    fluence_map_pri: npt.NDArray[np.float32],
    fluence_map_sec: npt.NDArray[np.float32],
    exam: Exam,
    jaw_x_positions: npt.NDArray[np.float32],
    jaw_y_positions: npt.NDArray[np.float32],
    settings: dict,
) -> npt.NDArray[np.float32]:
    """Run the CCC GPU dose calculation for a single plan/source configuration.

    Parameters
    ----------
    grid : Grid
        Target grid describing voxel geometry and providing ``num_voxels``,
        ``corner`` and ``resolution``. The grid ordering is (nx, ny, nz) for
        metadata while voxel arrays are allocated in (nz, ny, nx) shape.
    source : object
        Source descriptor with attributes required by the GPU kernels:
        ``position``, ``v_x``, ``v_y``, ``v_z`` and ``sad``.
    fluence_map_pri, fluence_map_sec : ndarray
        Primary and secondary fluence maps (precomputed or sampled) used by
        the fluence kernel.
    exam : Exam
        Exam object used to sample densities onto the target grid.
    jaw_x_positions, jaw_y_positions : ndarray
        Jaw positions in the X and Y axes (used when computing output factor
        correction).
    settings : Mapping[str, Any]
        Calculation settings and lookup tables used by kernels (energies,
        samples, off-axis softening tables, normalisation, etc.).

    Returns
    -------
    dose_grid : ndarray
        3D dose array (dtype float32) with shape (nz, ny, nx) representing
        calculated dose values in the grid's world units.
    """

    # Initialise the dose calculation kernel (precomputes kernel tables)
    kernel = Kernel(settings=settings)

    # Allocate GPU buffers
    oad_grid = np.zeros_like(grid.values, dtype=np.float32)
    d_geo_grid = np.zeros_like(grid.values, dtype=np.float32)
    d_eff_grid = np.zeros_like(grid.values, dtype=np.float32)
    density_grid = exam.densities_on_grid(grid).values
    fluence_grid = np.zeros_like(grid.values, dtype=np.float32)
    terma_grid = np.zeros_like(grid.values, dtype=np.float32)
    mask_grid = np.zeros_like(grid.values, dtype=np.float32)
    dose_grid = np.zeros_like(grid.values, dtype=np.float32)

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
        source_isocenter=source.isocenter,
    )
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0].imshow(oad_grid[:, :, 138], aspect='equal', cmap='jet')
    # axes[0].set_title("X slice")
    # axes[1].imshow(oad_grid[:, 103, :], aspect='equal', cmap='jet')
    # axes[1].set_title("Y slice")
    # axes[2].imshow(oad_grid[69, :, :], aspect='equal', cmap='jet')
    # axes[2].set_title("Z slice")
    # plt.tight_layout()
    # plt.show()
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
        source_isocenter=source.isocenter,
        source_sad=source.sad,
        pri_z=np.float32(settings["sources"]["pri_z"]),
        sec_z=np.float32(settings["sources"]["sec_z"]),
        samples=np.int32(settings["calculation"]["fluence_resampling"]),
    )


    # data = fluence_map_pri
    # rows, cols = data.shape
    # fig, ax = plt.subplots(figsize=(12,12))
    # # Define extent to align grid with pixel edges
    # extent = [0, cols, 0, rows]
    # ax.imshow(data, interpolation='none', origin='lower', cmap='rainbow')
    # ax.set_xticks(np.arange(0, cols, 1))
    # ax.set_yticks(np.arange(0, rows, 1))
    # # ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    # ax.grid(which='major', color='black', linestyle='-', linewidth=0.5)
    # # ax.tick_params(which='minor', size=0)
    # ax.set_xlim([260, 300])
    # ax.set_ylim([260, 300])
    # plt.show()

    # print(f"Source collimator angle: {source.collimator}")
    # fig, axes = plt.subplots(1, 3, figsize=(45, 15))
    # axes[0].imshow(fluence_grid[:, :, 100], aspect='equal', cmap='plasma')
    # axes[0].set_title("X slice")
    # data = fluence_grid[:, 0, :]
    # rows, cols = data.shape
    # axes[1].imshow(data, aspect='equal', cmap='rainbow')
    # axes[1].set_title("Y slice")
    # axes[1].set_xticks(np.arange(0, cols, 1))
    # axes[1].set_yticks(np.arange(0, rows, 1))
    # axes[1].grid(which='major', color='black', linestyle='-', linewidth=0.5)
    # axes[1].set_xlim([80, 120])
    # axes[1].set_ylim([80, 120])
    # axes[2].imshow(fluence_grid[100, :, :], aspect='equal', cmap='plasma')
    # axes[2].set_title("Z slice")
    # plt.tight_layout()
    # plt.show()



    # data = fluence_grid[:, :, 100]
    # rows, cols = data.shape
    # fig, ax = plt.subplots(figsize=(12,12))

    # # Define extent to align grid with pixel edges
    # extent = [0, cols, 0, rows]

    # ax.imshow(data, interpolation='none', origin='lower', cmap='rainbow')
    # ax.set_xticks(np.arange(0, cols, 1))
    # ax.set_yticks(np.arange(0, rows, 1))
    # # ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    # ax.grid(which='major', color='black', linestyle='-', linewidth=0.5)

    # # ax.tick_params(which='minor', size=0)
    # ax.set_xlim([0, 30])
    # ax.set_ylim([80, 120])
    # plt.show()








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
    x_gap = np.abs(jaw_x_positions[1] - jaw_x_positions[0])
    y_gap = np.abs(jaw_y_positions[1] - jaw_y_positions[0])
    equiv_square = 2 * x_gap * y_gap / (x_gap + y_gap)
    ofc = np.interp(
        equiv_square,
        settings["output_factor_correction"]["field_sizes"],
        settings["output_factor_correction"]["factors"],
    ).astype(np.float32)
    N = np.float32(settings["calculation"]["normalisation"])

    return dose_grid * ofc * N
