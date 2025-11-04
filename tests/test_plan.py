import os
import tempfile
import pytest
import pydicom.uid
from pydicom.dataset import FileDataset, FileMetaDataset, Dataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import toml

from conehead.plan import Plan, Beam, ControlPoint


def _make_rtplan_dataset():
    """Create a minimal in-memory RTPLAN-like dataset for testing."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.RTPlanStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    rt = FileDataset("rt.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    rt.SOPClassUID = pydicom.uid.RTPlanStorage

    # Basic patient/plan tags
    rt.RTPlanLabel = "TESTPLAN"
    rt.PatientName = "Doe^John"
    rt.PatientID = "P12345"
    rt.PatientBirthDate = "19800101"
    rt.PatientSex = "M"

    fgs = Dataset()
    fgs.FractionGroupNumber = 1
    rbs = Dataset()
    rbs.ReferencedBeamNumber = 1
    fgs.ReferencedBeamSequence = [rbs]
    rt.FractionGroupSequence = [fgs]

    # Build a simple BeamSequence with one beam and two control points
    b = Dataset()
    b.BeamNumber = 1
    b.BeamName = "Beam1"
    b.BeamType = "STATIC"
    b.IsocenterPosition = [0.0, 0.0, 0.0]

    # Control points
    cp0 = Dataset()
    cp0.GantryAngle = 0.0
    cp0.BeamLimitingDeviceAngle = 0.0
    cp0.PatientSupportAngle = 0.0

    # add a BeamLimitingDevicePositionSequence with LeafJawPositions
    bld0 = Dataset()
    bld0.RTBeamLimitingDeviceType = "ASYMX"
    bld0.LeafJawPositions = [-10.0, 10.0]
    cp0.BeamLimitingDevicePositionSequence = [bld0]

    cp1 = Dataset()
    cp1.GantryAngle = 90.0
    cp1.BeamLimitingDeviceAngle = 0.0
    cp1.PatientSupportAngle = 0.0
    bld1 = Dataset()
    bld1.RTBeamLimitingDeviceType = "ASYMX"
    bld1.LeafJawPositions = [-5.0, 5.0]
    cp1.BeamLimitingDevicePositionSequence = [bld1]

    b.ControlPointSequence = [cp0, cp1]

    rt.BeamSequence = [b]
    return rt


def test_plan_from_dataset_parses_basic_fields():
    ds = _make_rtplan_dataset()
    p = Plan(dataset=ds)

    assert p.plan_label == "TESTPLAN"
    assert p.patient_name is not None
    assert p.patient_id == "P12345"
    assert isinstance(p.beams, list)
    assert len(p.beams) == 1

    beam = p.beams[0]
    assert beam.number == 1
    assert beam.name == "Beam1"
    assert isinstance(beam.control_points, list)
    assert len(beam.control_points) == 2

    # control point fields
    cp = beam.control_points[0]
    assert isinstance(cp, ControlPoint)
    assert cp.gantry_angle == 0.0
    assert cp.x_jaw_positions is not None


def test_plan_from_path_and_summary(tmp_path):
    ds = _make_rtplan_dataset()
    pfile = tmp_path / "rtplan.dcm"
    ds.save_as(str(pfile), enforce_file_format=True)

    p = Plan(path=str(pfile))
    s = p.summary()
    assert s["plan_label"] in ("TESTPLAN", None)
    assert s["num_beams"] == 1
    assert s["beams"][0]["n_control_points"] == 2


def test_get_beam_controlpoint_structure():
    ds = _make_rtplan_dataset()
    p = Plan(dataset=ds)
    b = p.beams[0]
    # verify ControlPoint indexes are sequential and accessible
    assert [cp.index for cp in b.control_points] == [0, 1]
    # verify jaw positions captured from nested sequence
    j0 = b.control_points[0].x_jaw_positions
    assert isinstance(j0, list)
    # values are converted from mm to cm in the Plan loader
    assert pytest.approx(j0[0]) == -1.0
