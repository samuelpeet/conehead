from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import numpy as np
import numpy.typing as npt
import pydicom
from pydicom.dataset import Dataset as PydicomDataset

from conehead.grid import Grid


@dataclass
class ControlPoint:
    """Compact representation of a single control point in an RT Plan.

    Many RT Plan datasets contain vendor-specific and optional tags. This
    class captures the most commonly useful numeric parameters and keeps
    the original pydicom Dataset available for advanced inspection via
    the ``raw`` attribute.
    """

    index: int
    mlc_boundaries: npt.NDArray[np.float32]
    mlc_positions: npt.NDArray[np.float32]
    jaw_x_positions: npt.NDArray[np.float32]
    jaw_y_positions: npt.NDArray[np.float32]
    gantry: np.float32
    collimator: np.float32
    couch: np.float32
    cum_meterset_weight: np.float32
    diff_meterset_weight: np.float32
    wedge_angle: Optional[np.float32] = None
    wedge_orientation: Optional[np.float32] = None
    nominal_beam_energy: Optional[np.float32] = None
    isocenter_position: Optional[npt.NDArray[np.float32]] = None


@dataclass
class Beam:
    """Summary of a single beam within an RT Plan."""

    number: int
    name: str
    type: str
    mu: np.float32
    mlc_boundaries: npt.NDArray[np.float32]
    isocenter_position: npt.NDArray[np.float32]
    control_points: List[ControlPoint] = field(default_factory=list)
    dose: Optional[Grid] = None
    dose_spec_value: Optional[np.float32] = None
    dose_spec_point: Optional[npt.NDArray[np.float32]] = None


@dataclass
class Plan:
    """Simple loader for DICOM RT Plan files.

    Pass a path to directory holding an RT Plan DICOM file (or a pydicom Dataset) and the
    object will parse and expose a small set of commonly-used attributes:

    - ``plan_label``: RT Plan label / name
    - ``patient_name``, ``patient_id``, ``patient_birth_date``, ``patient_sex``
    - ``beams``: list of :class:`Beam` objects (each contains ControlPoints)

    The implementation uses safe getattr() lookups so it will tolerate
    incomplete or vendor-specific RT Plan files.
    """

    dicom_dir: Optional[str] = None
    dataset: Optional[PydicomDataset] = None
    plan_label: Optional[str] = None
    plan_date: Optional[str] = None
    plan_manufacturer: Optional[str] = None
    num_fractions: Optional[int] = None
    patient_position: Optional[str] = None
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    patient_birth_date: Optional[str] = None
    patient_sex: Optional[str] = None
    delivery_method: Optional[str] = None
    beams: List[Beam] = field(default_factory=list)
    dose: Optional[Grid] = None
    plan_instance_uid: Optional[str] = None

    def __post_init__(self) -> None:
        if self.dicom_dir and self.dataset is None:
            dicom_files = []
            for f in os.listdir(self.dicom_dir):
                if not f.endswith(".dcm"):
                    continue
                path = os.path.join(self.dicom_dir, f)
                try:
                    ds = pydicom.dcmread(path)
                except Exception:
                    # Skip files that fail to parse
                    print(f"Failed to read DICOM file: {path}")
                    continue
                dicom_files.append(ds)
            # Remove any non-RTPLAN files
            dicom_files = [
                f
                for f in dicom_files
                if getattr(f, "SOPClassUID", None) is not None
                and f.SOPClassUID.name == "RT Plan Storage"
            ]
            if len(dicom_files) == 0:
                # No RT Plan file found
                return None
            if len(dicom_files) > 1:
                raise ValueError(f"Multiple RT Plan DICOM files found in folder: {self.dicom_dir}")

            self.dataset = dicom_files[0]

        if self.dataset is not None:
            self._parse_dataset(self.dataset)

    def _parse_dataset(self, ds: PydicomDataset) -> None:
        # All numerical lists/values converted to np.float32 ndarrays.
        # Also converted to cm from DICOM mm.

        # Basic patient / plan metadata
        self.plan_label = getattr(ds, "RTPlanLabel", None) or getattr(ds, "RTPlanName", None)
        self.plan_date = getattr(ds, "RTPlanDate", None)
        self.plan_manufacturer = getattr(ds, "Manufacturer", None)
        self.delivery_method = getattr(ds, "TreatmentProtocols", None)
        self.plan_instance_uid = getattr(ds, "SOPInstanceUID", None)

        pn = getattr(ds, "PatientName", None)
        self.patient_name = str(pn) if pn is not None else None
        self.patient_id = getattr(ds, "PatientID", None)
        self.patient_birth_date = getattr(ds, "PatientBirthDate", None)
        self.patient_sex = getattr(ds, "PatientSex", None)

        # Patient setup information
        patient_pos_seq = getattr(ds, "PatientSetupSequence", None)
        if len(patient_pos_seq or []) > 1:
            raise NotImplementedError(
                "Multiple patient setups in PatientSetupSequence are not currently supported."
            )
        for pos in patient_pos_seq or []:
            self.patient_position = getattr(pos, "PatientPosition", None)
            if self.patient_position != "HFS":
                raise NotImplementedError(
                    "Only 'HFS' (Head First Supine) patient positions are currently supported."
                )

        # Beam (BeamSequence or RTBeamSequence in some vendors)
        beam_seq = getattr(ds, "BeamSequence", None) or getattr(ds, "RTBeamSequence", None) or []
        frac_group_seq = getattr(ds, "FractionGroupSequence", None) or []
        if frac_group_seq == [] or frac_group_seq is None:
            raise ValueError("FractionGroupSequence is missing from the RT Plan dataset.")
        self.num_fractions = getattr(ds, "NumberOfFractionsPlanned", None)
        ref_beam_seq = (
            getattr(frac_group_seq[0], "ReferencedBeamSequence", None)
            or getattr(frac_group_seq[0], "RTReferencedBeamSequence", None)
            or []
        )

        self.beams = []
        for b in beam_seq:
            number = getattr(b, "BeamNumber", None)
            if number is None:
                raise ValueError("BeamNumber is missing from a BeamSequence item.")
            name = getattr(b, "BeamName", None)
            if name is None:
                raise ValueError("BeamName is missing from a BeamSequence item.")
            type = getattr(b, "BeamType", None)
            if type is None:
                raise ValueError("BeamType is missing from a BeamSequence item.")

            # Default referenced values
            for r in ref_beam_seq:
                ref_number = getattr(r, "ReferencedBeamNumber", None)
                if ref_number == number:
                    # Found matching referenced beam
                    dose_spec_point = getattr(r, "BeamDoseSpecificationPoint", None)
                    if dose_spec_point is not None:
                        dose_spec_point = (
                            np.array(dose_spec_point, dtype=np.float32) * 0.1
                        )  # mm to cm
                    dose_spec_value = getattr(r, "BeamDose", None)
                    mu = getattr(r, "BeamMeterset", None)
                    if mu is None:
                        raise ValueError("BeamMeterset is missing from ReferencedBeamSequence.")
                    break

            # Get boundaries of MLCs
            for collimator in b.BeamLimitingDeviceSequence:
                if collimator.RTBeamLimitingDeviceType == "MLCX":
                    mlc_boundaries = collimator.LeafPositionBoundaries
                    mlc_boundaries = np.array(mlc_boundaries, dtype=np.float32) * 0.1  # mm to cm

            control_points = []
            # Control points live in ControlPointSequence
            cps = getattr(b, "ControlPointSequence", None) or []
            for i, cp in enumerate(cps):
                gantry = getattr(cp, "GantryAngle", None)
                collimator = getattr(cp, "BeamLimitingDeviceAngle", None)
                couch = getattr(cp, "PatientSupportAngle", None)
                if couch is not None and couch != 0:
                    raise NotImplementedError("Non-zero couch angles are not currently supported.")
                energy = getattr(cp, "NominalBeamEnergy", None) or getattr(
                    b, "NominalBeamEnergy", None
                )
                cum_meterset_weight = np.float32(getattr(cp, "CumulativeMetersetWeight", 0.0))
                if i == 0:
                    diff_meterset_weight = cum_meterset_weight
                else:
                    diff_meterset_weight = (
                        cum_meterset_weight - control_points[i - 1].cum_meterset_weight
                    )

                # Jaw positions and MLC positions
                bld_seq = getattr(cp, "BeamLimitingDevicePositionSequence", None)
                # initialize jaw/mlc containers per control point
                jaw_x_positions = None
                jaw_y_positions = None
                mlc_positions = None
                if bld_seq is not None:
                    for dev in bld_seq:
                        vals = []
                        pos = getattr(dev, "LeafJawPositions", None)
                        if pos is not None:
                            vals.extend(list(pos))
                        dev_type = getattr(dev, "RTBeamLimitingDeviceType", None)
                        if dev_type is None:
                            raise ValueError(
                                "RTBeamLimitingDeviceType is missing in BeamLimitingDevicePositionSequence"
                            )
                        elif dev_type == "ASYMX":
                            jaw_x_positions = vals
                        elif dev_type == "ASYMY":
                            jaw_y_positions = vals
                        elif dev_type == "MLCX":
                            mlc_positions = vals
                        else:
                            raise ValueError(
                                "Unknown RTBeamLimitingDeviceType in BeamLimitingDevicePositionSequence"
                            )

                # Handle wedge information if present
                wedge_angle = None
                wedge_orientation = None
                if hasattr(b, "NumberOfWedges"):
                    if b.NumberOfWedges == 1:
                        wedge_seq = getattr(b, "WedgeSequence")
                        wedge_type = getattr(wedge_seq[0], "WedgeType", None)
                        print(f"beam number {b.BeamNumber} wedge_type: {wedge_type}")
                        if wedge_type not in ["DYNAMIC"]:
                            raise ValueError(
                                f"Unsupported WedgeType {wedge_type} found in WedgeSequence"
                            )
                        wedge_angle = getattr(wedge_seq[0], "WedgeAngle", None)
                        if wedge_angle not in [10, 15, 20, 25, 30, 45, 60]:
                            raise ValueError(
                                f"Unsupported WedgeAngle {wedge_angle} found in WedgeSequence"
                            )
                        wedge_orientation = getattr(wedge_seq[0], "WedgeOrientation", None)
                        if wedge_orientation not in [0, 180]:
                            raise ValueError(
                                f"Unsupported WedgeOrientation {wedge_orientation} found in WedgeSequence"
                            )
                    elif b.NumberOfWedges > 1:
                        raise NotImplementedError(
                            "Multiple wedges per beam are not currently supported."
                        )

                isocenter_position = getattr(cp, "IsocenterPosition", None)

                # Unit conversion from mm to cm
                if isocenter_position is not None:
                    isocenter_position = np.array(isocenter_position, dtype=np.float32) * 0.1
                if jaw_x_positions is None:
                    # Enforce a default jaw position
                    jaw_x_positions = np.array([-20.0, 20.0], dtype=np.float32)  # default +/- 20 cm
                else:
                    jaw_x_positions = np.array(jaw_x_positions, dtype=np.float32) * 0.1
                if jaw_y_positions is None:
                    # Enforce a default jaw position
                    jaw_y_positions = np.array([-20.0, 20.0], dtype=np.float32)  # default +/- 20 cm
                else:
                    jaw_y_positions = np.array(jaw_y_positions, dtype=np.float32) * 0.1
                if mlc_positions is not None:
                    mlc_positions = np.array(mlc_positions, dtype=np.float32) * 0.1

                # Some attributes are only carried in the first control point. Propagate them.
                if gantry is None:
                    gantry = control_points[0].gantry
                if collimator is None:
                    collimator = control_points[0].collimator
                if couch is None:
                    couch = control_points[0].couch
                if energy is None:
                    energy = control_points[0].nominal_beam_energy
                if isocenter_position is None:
                    isocenter_position = control_points[0].isocenter_position
                # if wedge_angle is None:
                #     wedge_angle = control_points[0].wedge_angle
                # if wedge_orientation is None:
                #     wedge_orientation = control_points[0].wedge_orientation

                control = ControlPoint(
                    index=i,
                    gantry=np.float32(gantry),
                    collimator=np.float32(collimator),
                    couch=np.float32(couch),
                    cum_meterset_weight=cum_meterset_weight,
                    diff_meterset_weight=diff_meterset_weight,
                    wedge_angle=wedge_angle,
                    wedge_orientation=wedge_orientation,
                    nominal_beam_energy=energy,
                    jaw_x_positions=jaw_x_positions,
                    jaw_y_positions=jaw_y_positions,
                    mlc_boundaries=mlc_boundaries,
                    mlc_positions=mlc_positions,  # type: ignore
                    isocenter_position=isocenter_position,
                )
                control_points.append(control)

            beam = Beam(
                number=number,
                name=name,
                type=type,
                mu=np.float32(mu),
                dose_spec_value=np.float32(dose_spec_value),
                dose_spec_point=dose_spec_point,
                mlc_boundaries=mlc_boundaries,
                control_points=control_points,
                isocenter_position=control_points[0].isocenter_position,
            )
            self.beams.append(beam)

    def summary(self) -> Dict[str, Any]:
        """Return a compact summary dictionary of the loaded plan."""
        return {
            "plan_label": self.plan_label,
            "patient_name": self.patient_name,
            "patient_id": self.patient_id,
            "num_beams": len(self.beams),
            "beams": [
                {
                    "number": b.number,
                    "name": b.name,
                    "type": b.type,
                    "n_control_points": len(b.control_points),
                }
                for b in self.beams
            ],
        }
