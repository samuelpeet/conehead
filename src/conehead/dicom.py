"""Helpers for exporting dose arrays to RTDOSE DICOM files.

This module provides utilities to convert the project's internal dose
representation (attached to `Plan`/`Beam` objects) into RTDOSE
FileDatasets suitable for writing with :mod:`pydicom`.

The produced datasets populate the commonly-used RTDOSE attributes
required by downstream radiotherapy tools: study/series/patient
identifiers, grid geometry, DoseGridScaling and PixelData. UIDs are
generated where required and PixelData is stored as unsigned 16-bit
integers with an appropriate scaling factor so the original floating
point dose values can be recovered at read time.

Notes
-----
The helpers are intentionally lightweight: they do not attempt to be a
fully-featured DICOM writer. Callers must provide well-formed
``Plan`` and ``Exam`` objects whose attributes (patient identifiers,
beam dose arrays, geometric metadata) are valid for the intended
export.
"""

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset, Dataset
from pydicom.sequence import Sequence
from datetime import datetime, timezone
from typing import cast
from pydicom.uid import (
    ExplicitVRLittleEndian,
    RTDoseStorage,
    RTPlanStorage,
    generate_uid,
)
from conehead.plan import Plan
from conehead.exam import Exam
from conehead.grid import Grid


def export_dose(
    output_dir: str,
    plan: Plan,
    exam: Exam,
    info_in_file_name: bool = False,
    beam_doses: bool = False,
) -> None:
    """Export per-beam dose arrays from a Plan/Exam to RTDOSE DICOM files.

    Parameters
    ----------
    output_dir : str
        Directory where RTDOSE files will be written. The function composes
        filenames inside this directory.
    plan : :class:`conehead.plan.Plan`
        Parsed Plan object. Each beam in ``plan.beams`` is exported to a
        separate RTDOSE file.
    exam : :class:`conehead.exam.Exam`
        Exam object providing study-level metadata (study UID, patient
        identifiers, times, etc.).
    info_in_file_name : bool, optional
        If True the plan and/or beam name is used in the output filename; otherwise a
        UID-derived filename is used. Default is ``False``.
    beam_doses : bool, optional
        If True, also export doses for each beam individually.

    Returns
    -------
    None
        This function writes RTDOSE files to ``output_dir`` and returns
        nothing.

    Notes
    -----
    The dose values are rescaled to 16-bit unsigned integers, and the
    corresponding ``DoseGridScaling`` is computed and stored so that::

        original_dose = stored_pixel_value * DoseGridScaling

    The dataset is written using :meth:`pydicom.dataset.FileDataset.save_as`.
    """

    # Use UTC timestamps for created instances
    date_now = datetime.now(timezone.utc).strftime("%Y%m%d")
    time_now = datetime.now(timezone.utc).strftime("%H%M%S.%f")

    # Series/implementation/frame-of-reference UIDs are generated per
    # export call so multiple invocations do not collide.
    series_uid = generate_uid()
    implementation_uid = generate_uid()  # consider hardcoding per release

    # unique SOP Instance UID per exported RTDOSE
    instance_uid = generate_uid()

    # Sanitise plan name by replacing potentially problematic characters
    invalid_chars = '/\\?%*:|"<> '
    for char in invalid_chars:
        label = plan.plan_label.replace(char, "_")

    # Build filename
    if info_in_file_name:
        filename = f"{output_dir}/RD_{label}_Total.dcm"
    else:
        filename = f"{output_dir}/RD{instance_uid}.dcm"

    # Help the static analyzer: the beam.dose object is a Grid
    dose: Grid = cast(Grid, plan.dose)

    # ------------------------- File Meta -------------------------
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RTDoseStorage
    file_meta.MediaStorageSOPInstanceUID = instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = implementation_uid
    file_meta.ImplementationVersionName = "conehead"

    # Create the FileDataset (empty dataset body for now)
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # ---------------------- Standard attributes -------------------
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.InstanceCreationDate = date_now
    ds.InstanceCreationTime = time_now
    ds.SOPClassUID = RTDoseStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    # Study/series/time metadata come from the Exam object
    ds.StudyDate = exam.study_date
    ds.SeriesDate = date_now
    ds.ContentDate = date_now
    ds.StudyTime = exam.study_time
    ds.SeriesTime = time_now
    ds.ContentTime = time_now

    # Identification / manufacturer / patient fields
    ds.AccessionNumber = ""
    ds.Modality = "RTDOSE"
    ds.Manufacturer = "samuelpeet/conehead"
    ds.InstitutionName = ""
    ds.ReferringPhysicianName = ""
    ds.SeriesDescription = plan.plan_label
    ds.OperatorsName = ""
    ds.ManufacturerModelName = "conehead"

    ds.PatientName = plan.patient_name
    ds.PatientID = plan.patient_id
    ds.PatientBirthDate = plan.patient_birth_date
    ds.PatientSex = plan.patient_sex

    # -------------------- Geometry / Pixel attributes --------------
    # The internal `beam.dose` object is expected to expose
    # - num_voxels: (nx, ny, nz) voxel counts
    # - resolution: voxel sizes in cm (x,y,z)
    # - corner: image position (patient) of the first voxel in mm
    rows = int(dose.num_voxels[1])  # ny
    cols = int(dose.num_voxels[0])  # nx
    n_frames = int(dose.num_voxels[2])  # nz
    # voxel sizes in cm -> convert to mm
    dx_mm = float(dose.resolution[0]) * 10.0
    dy_mm = float(dose.resolution[1]) * 10.0
    dz_mm = float(dose.resolution[2]) * 10.0
    slice_thickness = dz_mm
    ds.SliceThickness = slice_thickness
    ds.SoftwareVersions = "alpha"

    ds.StudyInstanceUID = exam.study_instance_uid
    ds.SeriesInstanceUID = series_uid
    ds.StudyID = exam.study_id
    ds.SeriesNumber = "1"
    ds.InstanceNumber = str(1)
    ds.FrameOfReferenceUID = exam.frame_of_reference_uid

    # Spatial orientation/position
    # Our Grid.corner is the corner of the most-negative voxel (not the centre),
    # so ImagePositionPatient (which should point to the first voxel centre)
    # must add half a voxel in each axis before converting to mm.
    ip_x = float(dose.corner[0]) + 0.5 * float(dose.resolution[0])
    ip_y = float(dose.corner[1]) + 0.5 * float(dose.resolution[1])
    ip_z = float(dose.corner[2]) + 0.5 * float(dose.resolution[2])
    ds.ImagePositionPatient = [ip_x * 10.0, ip_y * 10.0, ip_z * 10.0]  # cm -> mm

    # Orientation: (row direction vector, column direction vector)
    # For an axis-aligned grid where +x corresponds to increasing column index
    # and +y to increasing row index:
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    # Pixel data descriptors
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.NumberOfFrames = str(n_frames)
    ds.FrameIncrementPointer = (0x3004, 0x000C)
    ds.Rows = rows
    ds.Columns = cols
    # PixelSpacing in mm: [row_spacing, column_spacing] per DICOM convention
    ds.PixelSpacing = [dy_mm, dx_mm]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0

    # ---------------------- RTDOSE-specific ------------------------
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseComment = ""
    ds.DoseSummationType = "PLAN"
    # GridFrameOffsetVector: offsets for each frame in mm
    # GridFrameOffsetVector: offsets (mm) for each frame measured from ImagePositionPatient.
    # Because ImagePositionPatient is the centre of frame 0, offsets are k * slice_thickness.
    ds.GridFrameOffsetVector = [float(k) * slice_thickness for k in range(n_frames)]
    ds.TissueHeterogeneityCorrection = ["IMAGE", "ROI_OVERRIDE"]

    # ---------------------- Referenced sequences ------------------
    # Referenced RT Plan Sequence -> references the RTPLAN SOP Instance
    ref_rt_plan_seq_item = Dataset()
    ref_rt_plan_seq_item.ReferencedSOPClassUID = RTPlanStorage
    ref_rt_plan_seq_item.ReferencedSOPInstanceUID = plan.plan_instance_uid
    ds.ReferencedRTPlanSequence = Sequence([ref_rt_plan_seq_item])

    # ---------------------- PixelData population ------------------
    # We rescale floating-point dose to 16-bit integers. The stored
    # pixel value v maps back to dose via Dose = v * DoseGridScaling.
    # Narrow Optional[np.ndarray] -> np.ndarray for static analyzers
    assert dose.values is not None, "dose.values must be populated before export"
    values = dose.values

    max_dose = float(values.max())
    if max_dose == 0:
        # Avoid division by zero: leave all zeros and set a default scaling
        arr = np.zeros((n_frames, rows, cols), dtype=np.uint16)
        scale_factor = 1.0
    else:
        # Normalize to [0, 65535]
        scaled = (values / max_dose) * 65535.0
        arr = scaled.astype(np.uint16)
        scale_factor = max_dose / 65535.0

    ds.PixelData = arr.tobytes()
    ds.DoseGridScaling = float(scale_factor)

    # Ensure file is written with explicit VR little endian
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Write file to disk
    ds.save_as(filename, enforce_file_format=True)
    print(f"Wrote RTDOSE file to: {filename}")

    if beam_doses:
        for i, beam in enumerate(plan.beams):
            # unique SOP Instance UID per exported RTDOSE
            instance_uid = generate_uid()

            # Sanitise plan name by replacing potentially problematic characters
            invalid_chars = '/\\?%*:|"<> '
            for char in invalid_chars:
                label = plan.plan_label.replace(char, "_")

            # Build filename
            if info_in_file_name and getattr(beam, "name", None):
                filename = f"{output_dir}/RD_{label}_{beam.name}.dcm"
            else:
                filename = f"{output_dir}/RD{instance_uid}.dcm"

            # Help the static analyzer: the beam.dose object is a Grid
            dose: Grid = cast(Grid, beam.dose)

            # ------------------------- File Meta -------------------------
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = RTDoseStorage
            file_meta.MediaStorageSOPInstanceUID = instance_uid
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            file_meta.ImplementationClassUID = implementation_uid
            file_meta.ImplementationVersionName = "conehead"

            # Create the FileDataset (empty dataset body for now)
            ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

            # ---------------------- Standard attributes -------------------
            ds.SpecificCharacterSet = "ISO_IR 100"
            ds.InstanceCreationDate = date_now
            ds.InstanceCreationTime = time_now
            ds.SOPClassUID = RTDoseStorage
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

            # Study/series/time metadata come from the Exam object
            ds.StudyDate = exam.study_date
            ds.SeriesDate = date_now
            ds.ContentDate = date_now
            ds.StudyTime = exam.study_time
            ds.SeriesTime = time_now
            ds.ContentTime = time_now

            # Identification / manufacturer / patient fields
            ds.AccessionNumber = ""
            ds.Modality = "RTDOSE"
            ds.Manufacturer = "samuelpeet/conehead"
            ds.InstitutionName = ""
            ds.ReferringPhysicianName = ""
            ds.SeriesDescription = plan.plan_label
            ds.OperatorsName = ""
            ds.ManufacturerModelName = "conehead"

            ds.PatientName = plan.patient_name
            ds.PatientID = plan.patient_id
            ds.PatientBirthDate = plan.patient_birth_date
            ds.PatientSex = plan.patient_sex

            # -------------------- Geometry / Pixel attributes --------------
            # The internal `beam.dose` object is expected to expose
            # - num_voxels: (nx, ny, nz) voxel counts
            # - resolution: voxel sizes in cm (x,y,z)
            # - corner: image position (patient) of the first voxel in mm
            rows = int(dose.num_voxels[1])  # ny
            cols = int(dose.num_voxels[0])  # nx
            n_frames = int(dose.num_voxels[2])  # nz
            # voxel sizes in cm -> convert to mm
            dx_mm = float(dose.resolution[0]) * 10.0
            dy_mm = float(dose.resolution[1]) * 10.0
            dz_mm = float(dose.resolution[2]) * 10.0
            slice_thickness = dz_mm
            ds.SliceThickness = slice_thickness
            ds.SoftwareVersions = "alpha"

            ds.StudyInstanceUID = exam.study_instance_uid
            ds.SeriesInstanceUID = series_uid
            ds.StudyID = exam.study_id
            ds.SeriesNumber = "1"
            ds.InstanceNumber = str(i + 1)
            ds.FrameOfReferenceUID = exam.frame_of_reference_uid


            # Spatial orientation/position
            # Our Grid.corner is the corner of the most-negative voxel (not the centre),
            # so ImagePositionPatient (which should point to the first voxel centre)
            # must add half a voxel in each axis before converting to mm.
            ip_x = float(dose.corner[0]) + 0.5 * float(dose.resolution[0])
            ip_y = float(dose.corner[1]) + 0.5 * float(dose.resolution[1])
            ip_z = float(dose.corner[2]) + 0.5 * float(dose.resolution[2])
            ds.ImagePositionPatient = [ip_x * 10.0, ip_y * 10.0, ip_z * 10.0]  # cm -> mm

            # Orientation: (row direction vector, column direction vector)
            # For an axis-aligned grid where +x corresponds to increasing column index
            # and +y to increasing row index:
            ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

            # Pixel data descriptors
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.NumberOfFrames = str(n_frames)
            ds.FrameIncrementPointer = (0x3004, 0x000C)
            ds.Rows = rows
            ds.Columns = cols
            # PixelSpacing in mm: [row_spacing, column_spacing] per DICOM convention
            ds.PixelSpacing = [dy_mm, dx_mm]
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0

            # ---------------------- RTDOSE-specific ------------------------
            ds.DoseUnits = "GY"
            ds.DoseType = "PHYSICAL"
            ds.DoseComment = ""
            ds.DoseSummationType = "BEAM"
            # GridFrameOffsetVector: offsets for each frame in mm
            # GridFrameOffsetVector: offsets (mm) for each frame measured from ImagePositionPatient.
            # Because ImagePositionPatient is the centre of frame 0, offsets are k * slice_thickness.
            ds.GridFrameOffsetVector = [float(k) * slice_thickness for k in range(n_frames)]
            ds.TissueHeterogeneityCorrection = ["IMAGE", "ROI_OVERRIDE"]

            # ---------------------- Referenced sequences ------------------
            # Referenced Beam Sequence -> used inside ReferencedFractionGroupSequence
            ref_beam_seq_item = Dataset()
            ref_beam_seq_item.ReferencedBeamNumber = beam.number
            ref_beam_seq = Sequence([ref_beam_seq_item])

            # Referenced Fraction Group Sequence -> includes referenced beams
            ref_frac_group_seq_item = Dataset()
            ref_frac_group_seq_item.ReferencedFractionGroupNumber = 1
            ref_frac_group_seq_item.ReferencedBeamSequence = ref_beam_seq
            ref_frac_group_seq = Sequence([ref_frac_group_seq_item])

            # Referenced RT Plan Sequence -> references the RTPLAN SOP Instance
            ref_rt_plan_seq_item = Dataset()
            ref_rt_plan_seq_item.ReferencedSOPClassUID = RTPlanStorage
            ref_rt_plan_seq_item.ReferencedSOPInstanceUID = plan.plan_instance_uid
            ref_rt_plan_seq_item.ReferencedFractionGroupSequence = ref_frac_group_seq
            ds.ReferencedRTPlanSequence = Sequence([ref_rt_plan_seq_item])

            # ---------------------- PixelData population ------------------
            # We rescale floating-point dose to 16-bit integers. The stored
            # pixel value v maps back to dose via Dose = v * DoseGridScaling.
            # Narrow Optional[np.ndarray] -> np.ndarray for static analyzers
            assert dose.values is not None, "dose.values must be populated before export"
            values = dose.values

            max_dose = float(values.max())
            if max_dose == 0:
                # Avoid division by zero: leave all zeros and set a default scaling
                arr = np.zeros((n_frames, rows, cols), dtype=np.uint16)
                scale_factor = 1.0
            else:
                # Normalize to [0, 65535]
                scaled = (values / max_dose) * 65535.0
                arr = scaled.astype(np.uint16)
                scale_factor = max_dose / 65535.0

            ds.PixelData = arr.tobytes()
            ds.DoseGridScaling = float(scale_factor)

            # Ensure file is written with explicit VR little endian
            ds.is_little_endian = True
            ds.is_implicit_VR = False

            # Write file to disk
            ds.save_as(filename, enforce_file_format=True)
            print(f"Wrote RTDOSE file to: {filename}")
