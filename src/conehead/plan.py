from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pydicom
from pydicom.dataset import Dataset as PydicomDataset


@dataclass
class ControlPoint:
    """Compact representation of a single control point in an RT Plan.

    Many RT Plan datasets contain vendor-specific and optional tags. This
    class captures the most commonly useful numeric parameters and keeps
    the original pydicom Dataset available for advanced inspection via
    the ``raw`` attribute.
    """

    index: int
    gantry_angle: Optional[float] = None
    collimator_angle: Optional[float] = None
    couch_angle: Optional[float] = None
    nominal_beam_energy: Optional[float] = None
    x_jaw_positions: Optional[List[float]] = None
    y_jaw_positions: Optional[List[float]] = None
    mlc_positions: Optional[Any] = None
    isocenter_position: Optional[List[float]] = None
    raw: Optional[PydicomDataset] = None


@dataclass
class Beam:
    """Summary of a single beam within an RT Plan."""

    number: Optional[int]
    name: Optional[str]
    type: Optional[str]
    mu: Optional[float] = None
    dose: Optional[float] = None
    dose_spec_point: Optional[List[float]] = None
    isocenter_position: Optional[List[float]] = None
    control_points: List[ControlPoint] = field(default_factory=list)
    raw: Optional[PydicomDataset] = None


@dataclass
class Plan:
    """Simple loader for DICOM RT Plan files.

    Pass a file path to an RT Plan DICOM file (or a pydicom Dataset) and the
    object will parse and expose a small set of commonly-used attributes:

    - ``plan_label``: RT Plan label / name
    - ``patient_name``, ``patient_id``, ``patient_birth_date``, ``patient_sex``
    - ``beams``: list of :class:`Beam` objects (each contains ControlPoints)

    The implementation uses safe getattr() lookups so it will tolerate
    incomplete or vendor-specific RT Plan files.
    """

    path: Optional[str] = None
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

    def __post_init__(self) -> None:
        if self.path and self.dataset is None:
            ds = pydicom.dcmread(self.path)
            self.dataset = ds

        if self.dataset is not None:
            self._parse_dataset(self.dataset)

    def _parse_dataset(self, ds: PydicomDataset) -> None:
        # Basic patient / plan metadata
        self.plan_label = getattr(ds, "RTPlanLabel", None) or getattr(ds, "RTPlanName", None)
        self.plan_date = getattr(ds, "RTPlanDate", None)
        self.plan_manufacturer = getattr(ds, "Manufacturer", None)
        self.delivery_method = getattr(ds, "TreatmentProtocols", None)

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
            name = getattr(b, "BeamName", None)
            btype = getattr(b, "BeamType", None)

            # default referenced values
            dose_spec_point = None
            dose = None
            mu = None
            for r in ref_beam_seq:
                ref_number = getattr(r, "ReferencedBeamNumber", None)
                if ref_number == number:
                    # Found matching referenced beam
                    dose_spec_point = getattr(r, "BeamDoseSpecificationPoint", None)
                    dose = getattr(r, "BeamDose", None)
                    mu = getattr(r, "BeamMeterset", None)
                    break

            beam = Beam(
                number=number,
                name=name,
                type=btype,
                mu=mu,
                dose=dose,
                dose_spec_point=dose_spec_point,
                raw=b,
            )

            # Control points live in ControlPointSequence
            cps = getattr(b, "ControlPointSequence", None) or []
            for i, cp in enumerate(cps):
                gantry = getattr(cp, "GantryAngle", None)
                coll = getattr(cp, "BeamLimitingDeviceAngle", None) or getattr(
                    cp, "CollimatorAngle", None
                )
                couch = getattr(cp, "PatientSupportAngle", None)
                if couch is not None and couch != 0:
                    raise NotImplementedError("Non-zero couch angles are not currently supported.")
                energy = getattr(cp, "NominalBeamEnergy", None) or getattr(
                    b, "NominalBeamEnergy", None
                )

                # Jaw positions and MLC positions
                bld_seq = getattr(cp, "BeamLimitingDevicePositionSequence", None)
                # initialize jaw/mlc containers per control point
                x_jaw_positions = None
                y_jaw_positions = None
                mlc_positions = None
                if bld_seq is not None:
                    for dev in bld_seq:
                        vals = []
                        pos = getattr(dev, "LeafJawPositions", None)
                        if pos is not None:
                            vals.extend(list(pos))
                        dev_type = getattr(dev, "RTBeamLimitingDeviceType", None)
                        if dev_type == "ASYMX":
                            x_jaw_positions = vals
                        elif dev_type == "ASYMY":
                            y_jaw_positions = vals
                        elif dev_type == "MLCX":
                            mlc_positions = vals
                        else:
                            raise ValueError(
                                "Unknown RTBeamLimitingDeviceType in BeamLimitingDevicePositionSequence"
                            )

                isocenter_position = getattr(cp, "IsocenterPosition", None)

                # Unit conversion from mm to cm
                if isocenter_position is not None:
                    isocenter_position = [x * 0.1 for x in isocenter_position]
                if x_jaw_positions is not None:
                    x_jaw_positions = [x * 0.1 for x in x_jaw_positions]
                if y_jaw_positions is not None:
                    y_jaw_positions = [x * 0.1 for x in y_jaw_positions]
                if mlc_positions is not None:
                    mlc_positions = [x * 0.1 for x in mlc_positions]

                # Keep raw cp for advanced inspection
                control = ControlPoint(
                    index=i,
                    gantry_angle=gantry,
                    collimator_angle=coll,
                    couch_angle=couch,
                    nominal_beam_energy=energy,
                    x_jaw_positions=x_jaw_positions,
                    y_jaw_positions=y_jaw_positions,
                    mlc_positions=mlc_positions,
                    isocenter_position=isocenter_position,
                    raw=cp,
                )
                beam.control_points.append(control)
                beam.isocenter_position = beam.control_points[0].isocenter_position

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
