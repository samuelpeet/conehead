# %%

# For dev
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../build/"))
import matplotlib.pyplot as plt


import numpy as np
import toml
from conehead.dicom import export_dose
from conehead.exam import Exam
from conehead.plan import Plan
from conehead.grid import Grid
from conehead.block import Block
from conehead.source import Source
from conehead.calculate import calculate


# Load machine settings
settings = toml.load("Truebeam_6FFF_M120.toml")

# Load CT/Structure Set and Plan
dicom_dir = "Prostate Wedge"
exam = Exam(dicom_dir=f"{dicom_dir}", hu_lut_path="Siemens_Confidence.toml")
plan = Plan(dicom_dir=f"{dicom_dir}")

# Define grid geometry for dose calculation
grid = Grid(
    corner=np.array([-26.97, -23.12, -10.20], dtype=np.float32),
    resolution=np.array([0.2, 0.2, 0.2], dtype=np.float32),
    num_voxels=np.array([267, 206, 138], dtype=np.int32),
)


for beam in plan.beams:
    beam.dose = Grid(corner=grid.corner, resolution=grid.resolution, num_voxels=grid.num_voxels)
    if beam.type == "STATIC":
        # This value means that all control point attributes in consecutive pairs of control points
        # are identical while only the cumulative meterset weight changes. This is typical for 3DCRT
        # and step-and-shoot IMRT plans. Step-and-shoot IMRT is not supported yet, so we will just
        # assume a STATIC beam means 3DCRT for now.

        # All the information we need is in the first control point.
        cp = beam.control_points[0]

        # Initialise the source geometry
        source = Source(isocenter=beam.isocenter_position)
        source.gantry = cp.gantry
        source.collimator = cp.collimator + 90.0
        print(f"Collimator, Gantry: {source.collimator}, {source.gantry}")
        print(f"Source position: {source.position}")
        print(f"Isocenter: {source.isocenter}")

        # Initalise the block and compute the fluence maps
        block = Block(control_point=cp, settings=settings)
        fluence_map_pri, fluence_map_sec = block.get_fluence_maps()

        # Pass everything through to the dose calculation function
        print(f"Calculating beam: {beam.name}")
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
        # Scale the dose by the beam MU and fraction number
        beam.dose.values *= beam.mu * plan.num_fractions  # type: ignore

    elif beam.type == "DYNAMIC":
        # This value means that multiple control point attributes change from one control point to
        # the next. This is typical of VMAT beams and sliding-window IMRT beams. IMRT is not
        # supported yet, so we will assume this is a VMAT beam for now.
        #
        # We will need to implement special handling for VMAT beams, as they typically contain many
        # control points with small angular increments. Performing a calculation at each control
        # point would be prohibitive, e.g., an arc with 180 control points (2 degree spacing) would
        # take 180 times longer to calculate than a 3DCRT beam at a single angle. So, we want to
        # sample the arc down to a manageable number of angles while still capturing the plan's
        # essence. We do this by grouping the control points into arc sectors of a user-specified
        # size (e.g., 10 degrees) and combining the control points within each sector into a single
        # representative control point by calculating an average fluence map weighted by the
        # relative meterset weight of each control point in the sector. We take the gantry angle at
        # the middle of the sector as the representative angle.

        # Let's start by inspecting the control points and grabbing the gantry angle at each point.
        # To handle the discontinuity at gantry 0/360, we map the gantry angles to a 0-360 degree
        # range starting from 6 o'clock and going clockwise.
        control_points = beam.control_points
        gantry_angles = [((float(cp.gantry) + 180.0) % 360.0) for cp in control_points]

        # Create the source and offset the dose grid as per the isocenter position


        # Now we want to iterate through the gantry angles and group the control points into sectors.
        # We start from the first angle and keep adding control points until we exceed the sector
        # angle limit.
        sectors = []
        sector_size = settings["calculation"]["arc_sector_size"]  # degrees
        current_sector_start_angle = gantry_angles[0]
        current_sector_cps = [control_points[0]]
        for i in range(1, len(gantry_angles)):
            angle = gantry_angles[i]
            if abs(angle - current_sector_start_angle) >= sector_size:
                # Finish the current sector
                sectors.append(current_sector_cps)
                # Start a new sector
                current_sector_start_angle = angle
                current_sector_cps = [control_points[i]]
            elif i == len(gantry_angles) - 1:
                # Last control point
                current_sector_cps.append(control_points[i])
                sectors.append(current_sector_cps)
            else:
                # Add to current sector
                current_sector_cps.append(control_points[i])

        # Now that we have the control points grouped into sectors, we combine the control points in
        # each sector to get a representative control point for that sector. We then calculate the
        # dose from this sector.
        for i, sector in enumerate(sectors):
            # Calculate the total meterset weight for the sector
            sector_meterset_weight = sector[-1].cum_meterset_weight - sector[0].cum_meterset_weight
            # Combine the control points in the sector into a single Block weighted by relative
            # meterset weight
            sector_block = Block(settings=settings)
            for cp in sector:
                cp_block = Block(settings=settings, control_point=cp)
                rel_weight = cp.diff_meterset_weight / sector_meterset_weight
                sector_block.values += cp_block.values * rel_weight
            fluence_map_pri, fluence_map_sec = sector_block.get_fluence_maps()

            # Calculate representative gantry angle for the sector (middle of start and end angles)
            sector_start_angle = (sector[0].gantry + 180.0) % 360.0
            sector_end_angle = (sector[-1].gantry + 180.0) % 360.0
            sector_mid_angle = (sector_start_angle + sector_end_angle) / 2
            sector_mid_angle = (sector_mid_angle + 180.0) % 360.0  # Map back to -180 to 180 range
            source = Source(isocenter=beam.isocenter_position)            
            source.gantry = np.float32(sector_mid_angle)
            source.collimator = np.float32(sector[0].collimator)

            # Calculate the average jaw positions for the sector for output factor correction
            x1 = np.mean([cp.jaw_x_positions[0] for cp in sector], dtype=np.float32)
            y1 = np.mean([cp.jaw_y_positions[0] for cp in sector], dtype=np.float32)
            x2 = np.mean([cp.jaw_x_positions[1] for cp in sector], dtype=np.float32)
            y2 = np.mean([cp.jaw_y_positions[1] for cp in sector], dtype=np.float32)
            jaw_x_positions = np.array([x1, x2], dtype=np.float32)
            jaw_y_positions = np.array([y1, y2], dtype=np.float32)

            # Pass everything through to the dose calculation function
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
            # The calculation for this sector is now complete.

        # Scale the dose by the beam MU and fraction number
        beam.dose.values *= beam.mu * plan.num_fractions  # type: ignore

# At this point, all beams have been processed and their dose distributions calculated.
# Let's sum the dose from all beams to get the total dose distribution for the plan.
plan.dose = Grid(corner=grid.corner, resolution=grid.resolution, num_voxels=grid.num_voxels)
for beam in plan.beams:
    if beam.dose is not None:
        plan.dose.values += beam.dose.values  # type: ignore

# Now let's export the total dose distribution to a DICOM RT Dose file.
export_dose(
    output_dir=dicom_dir,  # type: ignore
    plan=plan,
    exam=exam,
    info_in_file_name=True,
    beam_doses=True,
)

# Job's done!

# %%
