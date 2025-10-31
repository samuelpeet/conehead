import os
import toml
import numpy as np
import pydicom
import pydicom.uid

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
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.Rows = 2
    ds.Columns = 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.ImagePositionPatient = [0.0, 0.0, float(z)]
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    arr = np.ones((2, 2), dtype=np.uint16) * np.int16(value)
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
    _make_dicom_slice(dicom_dir / "slice0.dcm", z=0, value=100)
    _make_dicom_slice(dicom_dir / "slice1.dcm", z=1, value=200)

    ex = Exam(hu_lut_path=str(lut_path), dicom_folder=str(dicom_dir))
    assert hasattr(ex, "densities")
    g = ex.densities
    assert isinstance(g, Grid)
    # values should be float32 and have nz=2
    assert g.values.dtype == np.float32  # type: ignore
    assert g.values.shape[0] == 2  # type: ignore
