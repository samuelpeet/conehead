import numpy as np
import types
from conehead import runner


class FakeExam:
    def __init__(self, dicom_dir: str, hu_lut_path: str | None = None):
        self.dicom_dir = dicom_dir
        self.hu_lut_path = hu_lut_path


class FakeControlPoint:
    def __init__(
        self,
        gantry=0.0,
        collimator=0.0,
        jaw_x_positions=None,
        jaw_y_positions=None,
        cum_meterset_weight=1.0,
        diff_meterset_weight=1.0,
    ):
        self.gantry = gantry
        self.collimator = collimator
        self.jaw_x_positions = jaw_x_positions or np.array([-20.0, 20.0], dtype=np.float32)
        self.jaw_y_positions = jaw_y_positions or np.array([-20.0, 20.0], dtype=np.float32)
        self.cum_meterset_weight = cum_meterset_weight
        self.diff_meterset_weight = diff_meterset_weight
        self.nominal_beam_energy = None
        self.wedge_angle = None
        self.wedge_orientation = None
        self.isocenter_position = None


class FakeBeam:
    def __init__(self, name: str, btype: str = "STATIC"):
        self.number = 1
        self.name = name
        self.type = btype
        self.mu = np.float32(1.0)
        self.mlc_boundaries = np.array([-10.0, 10.0], dtype=np.float32)
        self.isocenter_position = None
        self.control_points = [FakeControlPoint()]


class FakePlan:
    def __init__(self, beams):
        self.beams = beams
        self.plan_label = "FAKE"


class FakeBlock:
    def __init__(self, *args, **kwargs):
        pass

    def get_fluence_maps(self):
        # return small fake fluence maps
        pri = np.ones((10, 10), dtype=np.float32)
        sec = np.zeros((10, 10), dtype=np.float32)
        return pri, sec


def test_run_plan_static_calls_calculate_and_export(monkeypatch, tmp_path):
    # Prepare fake settings loaded by toml.load
    fake_settings = {"calculation": {"arc_sector_size": 10.0}}
    monkeypatch.setattr(runner, "toml", types.SimpleNamespace(load=lambda p: fake_settings))

    # Replace Exam and Plan with fakes
    created_exams = []

    def exam_ctor(dicom_dir, hu_lut_path=None):
        e = FakeExam(dicom_dir, hu_lut_path)
        created_exams.append(e)
        return e

    monkeypatch.setattr(runner, "Exam", exam_ctor)

    fake_plan = FakePlan([FakeBeam("B1", "STATIC")])
    monkeypatch.setattr(runner, "Plan", lambda dicom_dir: fake_plan)

    # Patch Block and calculate and export_dose
    monkeypatch.setattr(runner, "Block", FakeBlock)

    called = {"calculate": 0}

    def calculate_mock(grid, source, pri, sec, exam, jx, jy, settings):
        called["calculate"] += 1
        # return an array matching the grid values shape
        return np.zeros_like(grid.values)

    monkeypatch.setattr(runner, "calculate", calculate_mock)

    export_calls = []

    def export_mock(**kwargs):
        export_calls.append(kwargs)

    monkeypatch.setattr(runner, "export_dose", export_mock)

    # Run
    out = str(tmp_path / "out")
    runner.run_plan(
        settings_path="settings.toml",
        dicom_dir="/fake/dicom",
        corner=(0.0, 0.0, 0.0),
        resolution=(1.0, 1.0, 1.0),
        num_voxels=(4, 4, 2),
        hu_lut_path="/path/to/hulut.toml",
        output_dir=out,
        beam_doses=True,
        info_in_file_name=False,
    )

    assert called["calculate"] >= 1
    assert len(export_calls) == 1
    # exam received hu_lut_path
    assert created_exams[0].hu_lut_path == "/path/to/hulut.toml"


def test_run_plan_dynamic_groups_and_sectoring(monkeypatch, tmp_path):
    fake_settings = {"calculation": {"arc_sector_size": 5.0}}
    monkeypatch.setattr(runner, "toml", types.SimpleNamespace(load=lambda p: fake_settings))

    monkeypatch.setattr(
        runner, "Exam", lambda dicom_dir, hu_lut_path=None: FakeExam(dicom_dir, hu_lut_path)
    )

    # create multiple control points with varying gantry to force sectoring
    cps = [FakeControlPoint(gantry=a) for a in [0, 3, 7, 12, 20]]
    beam = FakeBeam("VMAT", "DYNAMIC")
    beam.control_points = cps
    fake_plan = FakePlan([beam])
    monkeypatch.setattr(runner, "Plan", lambda dicom_dir: fake_plan)

    monkeypatch.setattr(runner, "Block", FakeBlock)

    calls = {"calculate": 0}

    def calculate_mock(grid, source, pri, sec, exam, jx, jy, settings):
        calls["calculate"] += 1
        return np.zeros_like(grid.values)

    monkeypatch.setattr(runner, "calculate", calculate_mock)
    monkeypatch.setattr(runner, "export_dose", lambda **kwargs: None)

    runner.run_plan(
        settings_path="settings.toml",
        dicom_dir="/fake/dicom",
        corner=(0.0, 0.0, 0.0),
        resolution=(1.0, 1.0, 1.0),
        num_voxels=(4, 4, 2),
        output_dir=str(tmp_path),
        hu_lut_path=None,
    )

    # For 5 control points and small sector size, calculate must be called at least once
    assert calls["calculate"] >= 1
