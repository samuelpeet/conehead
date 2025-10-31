# pyright: reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
import toml
import numpy as np
import pydicom
import pydicom.uid
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import pytest
import tempfile
import os

from conehead.grid import Grid
from conehead.exam import Exam


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

    ex = Exam(hu_lut_path=str(lut_path), dicom_folder=str(dicom_dir))
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
        # hu_lut_path is required when dicom_folder is provided
        Exam(dicom_folder=str(dicom_dir))


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
    ex = Exam(hu_lut_path=str(lut_path), dicom_folder=str(dicom_dir), densities=g)
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
        Exam(hu_lut_path=str(lpath), dicom_folder=str(dicom_dir))


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
        Exam(hu_lut_path=str(lpath), dicom_folder=str(dicom_dir))


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
        Exam(hu_lut_path=str(lpath), dicom_folder=str(dicom_dir))


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
        Exam(hu_lut_path=str(lpath), dicom_folder=str(dicom_dir))


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
