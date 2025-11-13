# pyright: reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
import toml
import numpy as np
import pydicom
import pydicom.uid
from pydicom.dataset import FileDataset, FileMetaDataset, Dataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import pytest
import tempfile
import os

from conehead.grid import Grid
from conehead.exam import Exam
from conehead.structure import ROI, StructureSet


def test_construct_with_densities_only():
    g = Grid(
        num_voxels=np.array([4, 3, 2], dtype=np.int32),
        corner=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        resolution=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )
    ex = Exam(densities=g)
    assert ex.densities is g
    assert getattr(ex, "hu_lut", None) is None


def test_construct_with_lut_only(tmp_path):
    lut = {"LUT": {"hu": [0, 1000], "density": [0.0, 1.0]}}
    p = tmp_path / "lut.toml"
    p.write_text(toml.dumps(lut))

    ex = Exam(hu_lut_path=str(p))
    assert isinstance(ex.hu_lut, dict)
    assert "hu" in ex.hu_lut and "density" in ex.hu_lut


def _make_dicom_slice(path, z, value=100):
    # Create a proper FileDataset with file meta so pydicom can round-trip

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.StudyInstanceUID = generate_uid()
    ds.StudyDate = "20240101"
    ds.StudyTime = "120000"
    ds.StudyID = "TESTSTUDY"
    ds.Rows = 3
    ds.Columns = 3
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.ImagePositionPatient = [0.0, 0.0, float(z)]
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    arr = np.ones((3, 3), dtype=np.int16) * np.int16(value)
    ds.PixelData = arr.tobytes()

    # Save using proper write semantics
    # Use the modern save argument to enforce a proper DICOM file format
    ds.save_as(str(path), enforce_file_format=True)


def test_construct_with_dicom_and_lut(tmp_path):
    # write LUT
    lut = {"LUT": {"hu": [0, 1000], "density": [0.0, 1.0]}}
    lut_path = tmp_path / "lut.toml"
    lut_path.write_text(toml.dumps(lut))

    dicom_dir = tmp_path / "dicoms"
    dicom_dir.mkdir()
    # create two simple slices at z=0 and z=1
    _make_dicom_slice(dicom_dir / "slice0.dcm", z=0, value=2)
    _make_dicom_slice(dicom_dir / "slice1.dcm", z=1, value=3)

    ex = Exam(hu_lut_path=str(lut_path), dicom_dir=str(dicom_dir))
    assert hasattr(ex, "densities")
    g = ex.densities
    assert isinstance(g, Grid)
    # values should be float32 and have nz=2, ny=3, nx=3
    assert g.values.dtype == np.float32  # type: ignore
    assert g.values.shape[0] == 2  # type: ignore
    assert g.values.shape[1] == 3  # type: ignore
    assert g.values.shape[2] == 3  # type: ignore


def test_dicom_requires_lut(tmp_path):
    dicom_dir = tmp_path / "dicoms"
    dicom_dir.mkdir()
    _make_dicom_slice(dicom_dir / "slice0.dcm", z=0, value=10)

    with pytest.raises(ValueError):
        # hu_lut_path is required when dicom_dir is provided
        Exam(dicom_dir=str(dicom_dir))


def test_invalid_lut_raises(tmp_path):
    # Write a TOML without a [LUT] table
    p = tmp_path / "bad_lut.toml"
    p.write_text(toml.dumps({"not_lut": {}}))

    with pytest.raises(ValueError):
        Exam(hu_lut_path=str(p))


def test_mismatched_lut_lengths_raise(tmp_path):
    lut = {"LUT": {"hu": [0, 100], "density": [0.0]}}
    p = tmp_path / "bad_lut2.toml"
    p.write_text(toml.dumps(lut))

    with pytest.raises(ValueError):
        Exam(hu_lut_path=str(p))


def test_densities_take_precedence_over_dicom(tmp_path):
    # Prepare a LUT and a dicom folder that would be loadable
    lut = {"LUT": {"hu": [0, 1000], "density": [0.0, 1.0]}}
    lut_path = tmp_path / "lut.toml"
    lut_path.write_text(toml.dumps(lut))

    dicom_dir = tmp_path / "dicoms2"
    dicom_dir.mkdir()
    _make_dicom_slice(dicom_dir / "slice0.dcm", z=0, value=2)

    # Provide densities explicitly; Exam should use them and not read DICOM
    g = Grid(num_voxels=[2, 2, 1], corner=[0.0, 0.0, 0.0], resolution=[1.0, 1.0, 1.0])
    ex = Exam(hu_lut_path=str(lut_path), dicom_dir=str(dicom_dir), densities=g)
    assert ex.densities is g


def test_plot_slice_and_densities_on_grid():
    # Create a small density Grid and check plot_slice runs and densities_on_grid
    # use a constant field so interpolation should reproduce the same constant
    vals = np.ones((2, 2, 2), dtype=np.float32) * 3.14
    g = Grid(num_voxels=[2, 2, 2], corner=[0.0, 0.0, 0.0], resolution=[1.0, 1.0, 1.0], values=vals)
    ex = Exam(densities=g)

    # plot_slice should not raise when given an axes
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ex.plot_slice(z_index=0, ax=ax)

    # Build a target grid whose voxel-centres match the original sampling
    # points used by the interpolator (original grid coordinates are at
    # corner + i*resolution; densities_on_grid samples at centres).
    target = Grid(num_voxels=[2, 2, 2], corner=[-0.5, -0.5, -0.5], resolution=[1.0, 1.0, 1.0])
    out = ex.densities_on_grid(target)
    assert isinstance(out, Grid)
    assert out.values.shape == g.values.shape
    assert np.allclose(out.values, g.values)


def test_constructor_rejects_non_grid_densities():
    with pytest.raises(TypeError):
        Exam(densities=object())


def test_hu_lut_missing_keys(tmp_path):
    # [LUT] present but missing 'hu' key
    p = tmp_path / "lut_missing_keys.toml"
    p.write_text(toml.dumps({"LUT": {"density": [0.0, 1.0]}}))
    with pytest.raises(ValueError):
        Exam(hu_lut_path=str(p))

    # [LUT] present but missing 'density' key
    p2 = tmp_path / "lut_missing_keys2.toml"
    p2.write_text(toml.dumps({"LUT": {"hu": [0, 100]}}))
    with pytest.raises(ValueError):
        Exam(hu_lut_path=str(p2))


def test_no_ct_files_found(tmp_path):
    # Create a DICOM file that is not a CT Image Storage

    dicom_dir = tmp_path / "nodct"
    dicom_dir.mkdir()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(dicom_dir / "s.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.Rows = 1
    ds.Columns = 1
    ds.PixelData = (np.ones((1, 1), dtype=np.uint8)).tobytes()
    ds.save_as(str(dicom_dir / "s.dcm"), enforce_file_format=True)

    lut = {"LUT": {"hu": [0, 100], "density": [0.0, 1.0]}}
    lpath = tmp_path / "lut2.toml"
    lpath.write_text(toml.dumps(lut))

    with pytest.raises(ValueError):
        Exam(hu_lut_path=str(lpath), dicom_dir=str(dicom_dir))


def test_missing_image_position_patient(tmp_path):
    # Create a DICOM missing ImagePositionPatient

    dicom_dir = tmp_path / "badpos"
    dicom_dir.mkdir()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(dicom_dir / "b.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.Rows = 1
    ds.Columns = 1
    ds.SliceThickness = 1.0
    ds.PixelSpacing = [1.0, 1.0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    # Intentionally omit ImagePositionPatient
    ds.PixelData = (np.ones((1, 1), dtype=np.uint8)).tobytes()
    ds.save_as(str(dicom_dir / "b.dcm"), enforce_file_format=True)

    lut = {"LUT": {"hu": [0, 100], "density": [0.0, 1.0]}}
    lpath = tmp_path / "lut3.toml"
    lpath.write_text(toml.dumps(lut))

    with pytest.raises(ValueError):
        Exam(hu_lut_path=str(lpath), dicom_dir=str(dicom_dir))


def test_missing_required_tag(tmp_path):
    # Create a DICOM missing 'Rows' tag

    dicom_dir = tmp_path / "badtag"
    dicom_dir.mkdir()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(dicom_dir / "t.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    # Intentionally omit Rows
    ds.Columns = 1
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.PixelData = (np.ones((1, 1), dtype=np.uint8)).tobytes()
    ds.save_as(str(dicom_dir / "t.dcm"), enforce_file_format=True)

    lut = {"LUT": {"hu": [0, 100], "density": [0.0, 1.0]}}
    lpath = tmp_path / "lut4.toml"
    lpath.write_text(toml.dumps(lut))

    with pytest.raises(ValueError):
        Exam(hu_lut_path=str(lpath), dicom_dir=str(dicom_dir))


def test_missing_pixel_array(tmp_path):
    # Create a DICOM that lacks PixelData so hasattr(ds, 'pixel_array') is False

    dicom_dir = tmp_path / "nopixel"
    dicom_dir.mkdir()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(dicom_dir / "n.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.Rows = 1
    ds.Columns = 1
    ds.SliceThickness = 1.0
    ds.PixelSpacing = [1.0, 1.0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    # Intentionally do not set PixelData
    ds.save_as(str(dicom_dir / "n.dcm"), enforce_file_format=True)

    lut = {"LUT": {"hu": [0, 100], "density": [0.0, 1.0]}}
    lpath = tmp_path / "lut5.toml"
    lpath.write_text(toml.dumps(lut))

    with pytest.raises(ValueError):
        Exam(hu_lut_path=str(lpath), dicom_dir=str(dicom_dir))


def test_runtime_error_when_lut_missing_in_loader():
    ex = Exam(densities=Grid(num_voxels=[1, 1, 1], corner=[0, 0, 0], resolution=[1, 1, 1]))
    # Create a temporary dicom folder with a minimal CT slice so the loader
    # proceeds far enough to hit the LUT-check and raise RuntimeError.
    td = tempfile.TemporaryDirectory()
    dpath = td.name
    # Use the existing helper to produce a well-formed CT slice
    _make_dicom_slice(os.path.join(dpath, "s.dcm"), z=0, value=1)

    with pytest.raises(RuntimeError):
        ex._load_dicom_series(dpath)
    td.cleanup()


def test_plot_slice_errors():
    ex = Exam()
    with pytest.raises(ValueError):
        ex.plot_slice()

    # Now set densities but remove values
    g = Grid(num_voxels=[1, 1, 1], corner=[0, 0, 0], resolution=[1, 1, 1])
    ex.densities = g
    ex.densities.values = None
    with pytest.raises(ValueError):
        ex.plot_slice()

    # Restore values and test index error
    ex.densities.values = np.zeros((1, 1, 1), dtype=np.float32)
    with pytest.raises(IndexError):
        ex.plot_slice(z_index=5)


def test_densities_on_grid_without_source():
    ex = Exam()
    target = Grid(num_voxels=[1, 1, 1], corner=[0, 0, 0], resolution=[1, 1, 1])
    with pytest.raises(Exception):
        ex.densities_on_grid(target)


def test_load_structures_and_masking(tmp_path):
    # Prepare LUT
    lut = {"LUT": {"hu": [0, 1000], "density": [0.0, 1.0]}}
    lut_path = tmp_path / "lut_struct.toml"
    lut_path.write_text(toml.dumps(lut))

    dicom_dir = tmp_path / "dicoms_struct"
    dicom_dir.mkdir()

    # Create two CT slices (z=0 and z=1)
    _make_dicom_slice(dicom_dir / "slice0.dcm", z=0, value=0)
    _make_dicom_slice(dicom_dir / "slice1.dcm", z=1, value=0)

    # Create an RTSTRUCT file with an External that covers whole space and
    # a small ROI with a density override.
    from pydicom.dataset import FileDataset, FileMetaDataset

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    rt = FileDataset(str(dicom_dir / "rt.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
    rt.SOPClassUID = pydicom.uid.RTStructureSetStorage

    # StructureSetROISequence: two ROIs (1=External, 2=Override)
    r1 = Dataset()
    r1.ROINumber = 1
    r1.ROIName = "External Body"
    r2 = Dataset()
    r2.ROINumber = 2
    r2.ROIName = "Override"
    rt.StructureSetROISequence = Sequence([r1, r2])

    # ROIContourSequence: External large contour and small override contour
    ext = Dataset()
    ext.ReferencedROINumber = 1
    ext_c = Dataset()
    # large square in mm that will cover the CT grid after conversion to cm
    ext_c.ContourData = [
        -100.0,
        -100.0,
        0.0,
        100.0,
        -100.0,
        0.0,
        100.0,
        100.0,
        0.0,
        -100.0,
        100.0,
        0.0,
    ]
    ext.ContourSequence = Sequence([ext_c])

    ov = Dataset()
    ov.ReferencedROINumber = 2
    ov_c = Dataset()
    # small square inside the first slice at mm coords
    ov_c.ContourData = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 5.0, 0.0]
    ov.ContourSequence = Sequence([ov_c])
    rt.ROIContourSequence = Sequence([ext, ov])

    # RTROIObservationsSequence: mark ROI 1 as EXTERNAL; ROI 2 has density override
    o1 = Dataset()
    o1.ReferencedROINumber = 1
    o1.RTROIInterpretedType = "EXTERNAL"
    o2 = Dataset()
    o2.ReferencedROINumber = 2
    prop = Dataset()
    prop.ROIPhysicalProperty = "REL_MASS_DENSITY"
    prop.ROIPhysicalPropertyValue = 2.5
    o2.ROIPhysicalPropertiesSequence = Sequence([prop])
    rt.RTROIObservationsSequence = Sequence([o1, o2])

    rt.save_as(str(dicom_dir / "rt.dcm"), enforce_file_format=True)

    # Now construct Exam which should load densities and structures
    ex = Exam(hu_lut_path=str(lut_path), dicom_dir=str(dicom_dir))
    assert hasattr(ex, "structure_set")
    assert ex.structure_set is not None
    assert hasattr(ex, "densities_masked")

    # External mask should flag outside voxels as -1.0
    ext_list = ex.structure_set.get_roi_by_type("EXTERNAL")
    assert len(ext_list) == 1
    external = ext_list[0]
    ext_mask = external.mask_on_grid(ex.densities)
    assert np.all(ex.densities_masked.values[~ext_mask] == -1.0)

    # Override ROI should set values to 2.5 where its mask is True
    roi2 = ex.structure_set.get_roi_by_number(2)
    assert roi2 is not None
    roi2_mask = roi2.mask_on_grid(ex.densities)
    # Only check that at least one masked voxel equals the override
    assert np.any(np.isclose(ex.densities_masked.values[roi2_mask], 2.5))


def test_load_structure_set_none_and_multiple(tmp_path):
    # When no RTSTRUCT present, _load_structure_set should return None
    dicom_dir = tmp_path / "nodct"
    dicom_dir.mkdir()
    _make_dicom_slice(dicom_dir / "slice0.dcm", z=0, value=1)

    ex = Exam(
        densities=Grid(
            num_voxels=np.array([1, 1, 1], dtype=np.int32),
            corner=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            resolution=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
    )
    res = ex._load_structure_set(str(dicom_dir))
    assert res is None

    # If multiple RT Structure files exist, a ValueError should be raised
    # Create two RT files
    from pydicom.dataset import FileDataset, FileMetaDataset

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    rt1 = FileDataset(str(dicom_dir / "r1.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
    rt1.SOPClassUID = pydicom.uid.RTStructureSetStorage
    rt1.StructureSetROISequence = Sequence([])
    rt1.save_as(str(dicom_dir / "r1.dcm"), enforce_file_format=True)

    rt2 = FileDataset(str(dicom_dir / "r2.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
    rt2.SOPClassUID = pydicom.uid.RTStructureSetStorage
    rt2.StructureSetROISequence = Sequence([])
    rt2.save_as(str(dicom_dir / "r2.dcm"), enforce_file_format=True)

    with pytest.raises(ValueError):
        ex._load_structure_set(str(dicom_dir))


def test_mask_raises_when_no_external():
    # Prepare a densities Grid and a StructureSet without an external ROI
    g = Grid(
        num_voxels=np.array([2, 2, 1], dtype=np.int32),
        corner=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        resolution=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )
    g.values[:] = 0.0
    ex = Exam(densities=g)

    # Create a StructureSet with an ROI that has no 'EXTERNAL' interpreted type
    roi = ROI(
        roi_number=1,
        name="SomeROI",
        contours=[np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [1.0, 1.0, 0.5]], dtype=np.float32)],
    )
    ss = StructureSet(rois=[roi])
    ex.structure_set = ss
    with pytest.raises(ValueError):
        ex._mask_densities_by_structures()
