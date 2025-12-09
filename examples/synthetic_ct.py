"""Generate a synthetic CT water phantom (cube) with surrounding air.

This script builds a DICOM CT series suitable for importing into a TPS.

Geometry
--------
Water cube size: 60 cm (x) × 60 cm (y) × 60 cm (z)
Air margin: 5 cm on all sides (total FOV 70 cm cube)
Slice thickness: 2 mm (⇒ 700 mm / 2 mm = 350 slices)
In-plane matrix: 512 × 512 (PixelSpacing ≈ 1.3672 mm)

Coordinate Convention
---------------------
Patient orientation: HFS (Head First Supine)
Patient coordinate axes (DICOM):
  x: increases toward patient left
  y: increases toward patient posterior
  z: increases toward patient superior

We place (0, 0, 0) at the water-air interface on the anterior face of the water
cube, centered laterally and vertically (x=0, y=0, z=0). Thus:
  Water extent in x: [-300 mm, +300 mm]
  Water extent in y: [0 mm, +600 mm]
  Water extent in z: [-300 mm, +300 mm]
Air fills the remaining volume within the 700 mm field:
  Total x extent: [-350 mm, +350 mm]
  Total y extent: [-50 mm, +650 mm]
  Total z extent: [-350 mm, +350 mm]

Pixel Classification
--------------------
Pixels with centres inside the water extents are assigned HU = 0 (water),
otherwise HU = -1000 (air). No smoothing (hard edge).

Required DICOM Tags
-------------------
We populate minimal, typical CT acquisition tags: Study/Series UIDs, SOP UIDs,
Patient identity placeholders, acquisition times, pixel geometry, rescale
parameters (slope=1, intercept=0), and CT-specific descriptors.

Usage
-----
Run directly:
	python synthetic_ct.py --output-dir examples/synthetic_ct

This creates 350 axial DICOM slices named CT_0001.dcm ... CT_0350.dcm.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian


def generate_volume() -> np.ndarray:
	"""Generate the HU volume (shape (slices, rows, cols)).

	Returns
	-------
	np.ndarray (int16): HU values, signed 16-bit.
	"""
	# Constants (mm)
	fov_mm = 700.0  # total field of view each axis
	water_half_mm = 300.0
	air_margin_mm = 50.0  # margin each side (since 60cm cube + 5cm each side => 70cm total)
	pixel_spacing_mm = fov_mm / 512.0  # ≈ 1.3672
	slice_thickness_mm = 2.0
	n_slices = int(round(fov_mm / slice_thickness_mm))  # 350

	# Coordinate arrays for pixel centres
	x0 = -fov_mm / 2.0 + pixel_spacing_mm / 2.0
	y0 = -air_margin_mm + pixel_spacing_mm / 2.0
	z0 = -fov_mm / 2.0 + slice_thickness_mm / 2.0
	xs = x0 + pixel_spacing_mm * np.arange(512)
	ys = y0 + pixel_spacing_mm * np.arange(512)
	zs = z0 + slice_thickness_mm * np.arange(n_slices)

	# Determine water extents
	water_x_min, water_x_max = -water_half_mm, water_half_mm
	water_y_min, water_y_max = 0.0, 600.0
	water_z_min, water_z_max = -water_half_mm, water_half_mm

	# Prepare empty volume (air HU = -1000)
	vol = np.full((n_slices, 512, 512), -1000, dtype=np.int16)

	# Create boolean masks for water region
	x_mask = (xs >= water_x_min) & (xs <= water_x_max)
	y_mask = (ys >= water_y_min) & (ys <= water_y_max)
	z_mask = (zs >= water_z_min) & (zs <= water_z_max)

	# Iterate slices; vectorise in-plane
	for iz, z_in_water in enumerate(z_mask):
		if not z_in_water:
			continue
		# 2D mask for water in this slice
		water_mask_2d = np.outer(y_mask, x_mask)  # (512,512) with rows=y, cols=x
		vol[iz, water_mask_2d] = 0  # water HU

	return vol


def create_ct_slice(hus: np.ndarray, slice_index: int, total_slices: int, study_uid: str, series_uid: str,
					frame_of_reference_uid: str,
					pixel_spacing_mm: float, slice_thickness_mm: float, z_position_mm: float,
					dt: datetime, patient_name: str, patient_id: str) -> FileDataset:
	"""Create a single CT DICOM slice dataset."""
	rows, cols = hus.shape
	file_meta = pydicom.dataset.FileMetaDataset()
	file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
	file_meta.MediaStorageSOPInstanceUID = generate_uid()
	file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
	file_meta.ImplementationClassUID = generate_uid()

	ds = FileDataset(f"CT_{slice_index:04d}.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)

	# Core identifiers
	ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
	ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
	ds.StudyInstanceUID = study_uid
	ds.SeriesInstanceUID = series_uid
	ds.Modality = "CT"
	ds.PatientName = patient_name
	ds.PatientID = patient_id
	ds.PatientBirthDate = "19700101"
	ds.PatientSex = "O"
	ds.StudyDate = dt.strftime("%Y%m%d")
	ds.StudyTime = dt.strftime("%H%M%S")
	ds.SeriesDate = ds.StudyDate
	ds.SeriesTime = ds.StudyTime
	ds.AccessionNumber = "SYNCT"  # placeholder
	ds.Manufacturer = "Synthetic"
	ds.InstitutionName = "SyntheticPhantomLab"
	ds.ReferringPhysicianName = "Phantom^Builder"
	ds.StudyDescription = "Synthetic Water Phantom"
	ds.SeriesDescription = "Synthetic Water Cube"
	ds.BodyPartExamined = "PHANTOM"
	ds.PatientPosition = "HFS"

	# Geometry
	ds.Rows = rows
	ds.Columns = cols
	ds.PixelSpacing = [str(pixel_spacing_mm), str(pixel_spacing_mm)]
	ds.SliceThickness = str(slice_thickness_mm)
	ds.SpacingBetweenSlices = str(slice_thickness_mm)
	ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
	# Compute ImagePositionPatient for first pixel (row0,col0) of this slice
	# Consistent with generate_volume(): x0,y0 and variable z
	fov_mm = 700.0
	x0 = -fov_mm / 2.0 + pixel_spacing_mm / 2.0
	y0 = -50.0 + pixel_spacing_mm / 2.0  # -50mm anterior air margin
	ds.ImagePositionPatient = [str(x0), str(y0), str(z_position_mm)]
	ds.InstanceNumber = slice_index + 1
	ds.GantryDetectorTilt = "0"
	ds.AcquisitionNumber = 1
	ds.KVP = "120"
	ds.XRayTubeCurrent = "500"
	ds.Exposure = "10"
	ds.FilterType = "NONE"
	ds.ConvolutionKernel = "STANDARD"
	ds.ReconstructionDiameter = str(fov_mm)

	# Pixel data characteristics
	ds.SamplesPerPixel = 1
	ds.PhotometricInterpretation = "MONOCHROME2"
	ds.BitsAllocated = 16
	ds.BitsStored = 16
	ds.HighBit = 15
	ds.PixelRepresentation = 1  # signed
	ds.RescaleIntercept = 0
	ds.RescaleSlope = 1
	ds.RescaleType = "HU"
	ds.WindowCenter = "0"
	ds.WindowWidth = "1000"

	# Convert pixel array to bytes
	ds.PixelData = hus.astype(np.int16).tobytes()

	# Derivations / frame references
	ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
	ds.FrameOfReferenceUID = frame_of_reference_uid
	ds.PositionReferenceIndicator = "SN"

	# Slice location (approximate): use z centre of slice plane
	ds.SliceLocation = str(z_position_mm)

	return ds


def write_series(output_dir: str, patient_name: str = "WATER^PHANTOM", patient_id: str = "WATER001") -> None:
	os.makedirs(output_dir, exist_ok=True)
	dt = datetime.now()
	study_uid = generate_uid()
	series_uid = generate_uid()

	vol = generate_volume()  # (n_slices, 512, 512)
	frame_of_reference_uid = generate_uid()
	n_slices = vol.shape[0]
	pixel_spacing_mm = 700.0 / 512.0
	slice_thickness_mm = 2.0
	# z positions for ImagePositionPatient (pixel centres plane origin)
	z0 = -700.0 / 2.0 + slice_thickness_mm / 2.0
	z_positions = z0 + slice_thickness_mm * np.arange(n_slices)

	for idx in range(n_slices):
		ds = create_ct_slice(
			vol[idx], idx, n_slices, study_uid, series_uid, frame_of_reference_uid,
			pixel_spacing_mm, slice_thickness_mm, z_positions[idx],
			dt, patient_name, patient_id,
		)
		fname = os.path.join(output_dir, f"CT_{idx+1:04d}.dcm")
		ds.save_as(fname, write_like_original=False)

	print(f"Wrote {n_slices} CT slices to {output_dir}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate synthetic water phantom CT series")
	parser.add_argument("--output-dir", default="examples/synthetic_ct", help="Destination directory for DICOM slices")
	parser.add_argument("--patient-name", default="WATER^PHANTOM", help="PatientName tag value")
	parser.add_argument("--patient-id", default="WATER001", help="PatientID tag value")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	write_series(args.output_dir, patient_name=args.patient_name, patient_id=args.patient_id)


if __name__ == "__main__":
	main()

