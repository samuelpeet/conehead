"""High-level runner to compute plan dose and export RTDOSE files.

This module provides a single entrypoint `run_plan` which packages the
logic from `examples/plan_import.py` into a reusable function (callable
from tests, notebooks or a CLI wrapper).

Notes
-----
- Grid corner/resolution are expected in centimetres.
- This function focuses on orchestration: loading settings, exam,
  plan, building the Grid, computing per-beam dose and exporting via
  `export_dose`.
"""

from __future__ import annotations
import logging
from typing import Optional, Sequence
import numpy as np
import toml
from conehead.exam import Exam
from conehead.plan import Plan
from conehead.grid import Grid
from conehead.block import Block
from conehead.source import Source
from conehead.calculate import calculate
from conehead.dicom import export_dose


logger = logging.getLogger(__name__)


def run_plan(
    settings_path: str,
    dicom_dir: str,
    corner: Sequence[float],
    resolution: Sequence[float],
    num_voxels: Sequence[int],
    hu_lut_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    info_in_file_name: bool = False,
    beam_doses: bool = False,
) -> None:
    """Run dose calculation for the RT plan(s) found in ``dicom_dir``.

    Parameters
    ----------
    settings_path
        Path to the machine settings TOML file.
    dicom_dir
        Directory containing CT/RTPLAN/RTSTRUCT DICOM files.
    corner
        Grid corner in cm (x, y, z) — most-negative-corner convention.
    resolution
        Voxel size in cm (dx, dy, dz).
    num_voxels
        Number of voxels (nx, ny, nz).
    output_dir
        Where to write RTDOSE files. If None, defaults to ``dicom_dir``.
    beam_doses
        If True, write per-beam RTDOSE files.
    info_in_file_name
        If True, add plan/beam info to generated filenames.
    save_plan_sum
        If True, also save a single RTDOSE with the plan-summed dose.
    """
    settings = toml.load(settings_path)
    out_dir = output_dir or dicom_dir

    logger.info("Loading exam from %s", dicom_dir)
    exam = Exam(dicom_dir=dicom_dir, hu_lut_path=hu_lut_path)

    logger.info("Loading plan from %s", dicom_dir)
    plan = Plan(dicom_dir=dicom_dir)

    # Build the top-level grid used for plan accumulation
    grid = Grid(
        corner=np.array(corner, dtype=np.float32),
        resolution=np.array(resolution, dtype=np.float32),
        num_voxels=np.array(num_voxels, dtype=np.int32),
    )

    # Compute per-beam dose
    for beam in plan.beams:
        logger.info("Computing dose for beam %s (%s)", beam.name, beam.type)
        beam.dose = Grid(corner=grid.corner, resolution=grid.resolution, num_voxels=grid.num_voxels)

        if beam.type == "STATIC":
            cp = beam.control_points[0]
            source = Source()
            source.gantry = cp.gantry
            source.collimator = cp.collimator

            block = Block(control_point=cp, settings=settings)
            fluence_map_pri, fluence_map_sec = block.get_fluence_maps()

            beam.dose.values += calculate(  # type: ignore
                grid,
                source,
                fluence_map_pri,
                fluence_map_sec,
                exam,
                cp.jaw_x_positions,
                cp.jaw_y_positions,
                settings,
            )
            beam.dose.values *= beam.mu  # type: ignore

        elif beam.type == "DYNAMIC":
            # VMAT: group control points into sectors to keep compute tractable.
            control_points = beam.control_points
            gantry_angles = [((float(cp.gantry) + 180.0) % 360.0) for cp in control_points]

            sector_size = settings.get("calculation", {}).get("arc_sector_size", 10.0)
            sectors = []
            current_sector_start = gantry_angles[0]
            current_sector_cps = [control_points[0]]

            for i in range(1, len(gantry_angles)):
                angle = gantry_angles[i]
                if abs(angle - current_sector_start) >= sector_size:
                    sectors.append(current_sector_cps)
                    current_sector_start = angle
                    current_sector_cps = [control_points[i]]
                else:
                    current_sector_cps.append(control_points[i])
            if current_sector_cps:
                sectors.append(current_sector_cps)

            for sector in sectors:
                sector_mw = sector[-1].cum_meterset_weight - sector[0].cum_meterset_weight
                if sector_mw == 0:
                    # Avoid division by zero; skip sectors with zero meterset
                    continue
                # Build a sector-averaged block
                sector_block = Block(settings=settings)
                for cp in sector:
                    cp_block = Block(settings=settings, control_point=cp)
                    rel_weight = float(cp.diff_meterset_weight) / float(sector_mw)
                    sector_block.values += cp_block.values * rel_weight
                fluence_map_pri, fluence_map_sec = sector_block.get_fluence_maps()

                sector_start_angle = (sector[0].gantry + 180.0) % 360.0
                sector_end_angle = (sector[-1].gantry + 180.0) % 360.0
                sector_mid_angle = (sector_start_angle + sector_end_angle) / 2.0
                sector_mid_angle = (sector_mid_angle + 180.0) % 360.0

                source = Source()
                source.gantry = np.float32(sector_mid_angle)
                source.collimator = np.float32(sector[0].collimator)

                x1 = np.mean([cp.jaw_x_positions[0] for cp in sector], dtype=np.float32)
                y1 = np.mean([cp.jaw_y_positions[0] for cp in sector], dtype=np.float32)
                x2 = np.mean([cp.jaw_x_positions[1] for cp in sector], dtype=np.float32)
                y2 = np.mean([cp.jaw_y_positions[1] for cp in sector], dtype=np.float32)
                jaw_x_positions = np.array([x1, x2], dtype=np.float32)
                jaw_y_positions = np.array([y1, y2], dtype=np.float32)

                beam.dose.values += calculate(  # type: ignore
                    grid,
                    source,
                    fluence_map_pri,
                    fluence_map_sec,
                    exam,
                    jaw_x_positions,
                    jaw_y_positions,
                    settings,
                )
            beam.dose.values *= beam.mu  # type: ignore
        else:
            logger.warning("Unknown beam type %r; skipping dose calc", beam.type)

    # Sum to plan dose
    plan.dose = Grid(corner=grid.corner, resolution=grid.resolution, num_voxels=grid.num_voxels)
    for beam in plan.beams:
        if getattr(beam, "dose", None) is not None:
            plan.dose.values += beam.dose.values  # type: ignore

    # Export — keep parameter names compatible with existing example usage
    export_dose(
        output_dir=out_dir,
        plan=plan,
        exam=exam,
        info_in_file_name=info_in_file_name,
        beam_doses=beam_doses,
    )
