"""RT Structure Set utilities.

This module provides small helpers to parse DICOM RT Structure Set
(RTSTRUCT) datasets into lightweight Python objects and to rasterise
ROIs onto project Grids.

Notes
-----
- Coordinates parsed from DICOM contours are by default converted from
    millimetres to centimetres (see ``ROI.from_dicom_items`` ``convert_to_cm``
    argument) so they match the repository convention where spatial units
    are centimetres.
- The rasterisation routine ``ROI.mask_on_grid`` uses OpenCV's
    ``fillPoly`` on each axial contour and therefore requires ``opencv-python``
    to be available in the runtime environment.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from pydicom.dataset import Dataset as PydicomDataset
import cv2
from conehead.grid import Grid


@dataclass
class ROI:
    """Representation of a single ROI (structure) from an RT Structure Set.

    Attributes
    ----------
    roi_number
        Integer ROI number (DICOM tag (3006,0022)).
    name
        Human-readable ROI name (ROIName tag).
    roi_type
        Interpreted ROI type where available (e.g. 'EXTERNAL', 'ORGAN').
    colour
        Optional RGB colour triplet as a list of three integers (0-255)
    contours
        List of numpy arrays, each with shape (N, 3), containing the XYZ
        contour points for each contour in the ROI. By default points are
        converted to centimetres (cm) to match project units; see
        ``convert_to_cm``.
    density
        Optional mass density override value (g/cm³) if specified in the RTSTRUCT.
    """

    roi_number: int
    name: str
    roi_type: Optional[str] = None
    colour: Optional[List[int]] = None
    contours: List[np.ndarray] = field(default_factory=list)
    density: Optional[float] = None

    @classmethod
    def from_dicom_items(
        cls,
        def_item: PydicomDataset,
        contour_item: Optional[PydicomDataset],
        obs_item: Optional[PydicomDataset],
        convert_to_cm: bool = True,
    ) -> ROI:
        """Construct an ROI from DICOM Sequence items.

        Parameters
        ----------
        def_item
            Item from StructureSetROISequence (contains ROINumber, ROIName).
        contour_item
            Corresponding item from ROIContourSequence (may be None if no
            contours present).
        obs_item
            Corresponding item from RTROIObservationsSequence (may be None).
        convert_to_cm
            Convert coordinates from mm to cm (multiply by 0.1) when True.
        """
        roi_number = int(getattr(def_item, "ROINumber"))
        name = getattr(def_item, "ROIName", "")
        roi_type = getattr(obs_item, "RTROIInterpretedType", None)

        colour = None
        if contour_item is not None and hasattr(contour_item, "ROIDisplayColor"):
            colour = getattr(contour_item, "ROIDisplayColor", None)

        contours: List[np.ndarray] = []
        if contour_item is not None and hasattr(contour_item, "ContourSequence"):
            for c in contour_item.ContourSequence:
                data = getattr(c, "ContourData", None)
                if data is None:
                    continue
                # ContourData is a flat list [x1,y1,z1, x2,y2,z2, ...]
                arr = np.asarray(data, dtype=np.float64).reshape(-1, 3)
                if convert_to_cm:
                    arr = arr * 0.1
                contours.append(arr.astype(np.float32))

        density = None
        if obs_item is not None and hasattr(obs_item, "ROIPhysicalPropertiesSequence"):
            for prop in obs_item.ROIPhysicalPropertiesSequence:
                if getattr(prop, "ROIPhysicalProperty", "").upper() == "REL_MASS_DENSITY":
                    density = float(getattr(prop, "ROIPhysicalPropertyValue"))
                    break

        return cls(
            roi_number=roi_number,
            name=name,
            roi_type=roi_type,
            colour=colour,
            contours=contours,
            density=density,
        )

    def mask_on_grid(self, grid: Grid) -> np.ndarray:
        """Rasterise the ROI onto the provided ``grid`` and return a boolean
        mask with shape ``(nz, ny, nx)``.

        Each contour is assumed to lie on a single axial (constant-Z) plane;
        the method finds the nearest Z-slice in ``grid`` and uses OpenCV's
        ``fillPoly`` to rasterise the 2D polygon into that slice. The ROI's
        coordinates are expected to be in the same units as ``grid`` (the
        repository convention is centimetres when DICOM -> ROI conversion is
        performed with ``convert_to_cm=True``).

        Parameters
        ----------
        grid
            Target :class:`Grid` describing the volume to rasterise into.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape (nz, ny, nx) where voxels inside the ROI
            are True.
        """
        # Grid dimensions: num_voxels stored as (nx, ny, nz) but values
        # have shape (nz, ny, nx).
        nx, ny, nz = int(grid.num_voxels[0]), int(grid.num_voxels[1]), int(grid.num_voxels[2])

        mask = np.zeros((nz, ny, nx), dtype=bool)

        # Grid world origin and voxel sizes (assumed same units as contours)
        cx, cy, cz = float(grid.corner[0]), float(grid.corner[1]), float(grid.corner[2])
        dx, dy, dz = float(grid.resolution[0]), float(grid.resolution[1]), float(grid.resolution[2])

        for contour in self.contours:
            if contour is None or contour.size == 0:
                continue

            # Use the mean Z for the contour; many RTSTRUCT contours are
            # planar so this is stable. If the contour spans multiple Z
            # values we still use the mean as a best-effort placement.
            zs = contour[:, 2]
            z_mean = float(np.mean(zs))

            # Map world-Z to nearest slice index. Use the same voxel-centre
            # convention as for X/Y: voxel-centre at corner + (i+0.5)*res.
            zf = (z_mean - cz) / dz - 0.5
            z_idx = int(np.round(zf))
            if z_idx < 0 or z_idx >= nz:
                # Contour lies outside provided grid
                continue

            # Map contour XY to image pixel coordinates. For a voxel index
            # ix/iy the voxel-centre is at corner + (ix+0.5)*resolution, so
            # converting world->index we subtract 0.5 to map centres to
            # integer indices.
            xs = contour[:, 0]
            ys = contour[:, 1]
            cols = (xs - cx) / dx - 0.5
            rows = (ys - cy) / dy - 0.5

            pts = np.vstack([cols, rows]).T
            if pts.shape[0] < 3:
                # Not a polygon
                continue

            pts_int = np.round(pts).astype(np.int32)

            # Create a temporary 2D slice image (height=ny, width=nx)
            slice_img = np.zeros((ny, nx), dtype=np.uint8)

            # OpenCV uses (x, y) == (col, row) ordering and expects an
            # array of shape (n_points, 1, 2).
            poly = pts_int.reshape((-1, 1, 2))

            try:
                cv2.fillPoly(slice_img, [poly], color=1)
            except Exception:
                # Be conservative: ensure int32 and retry
                cv2.fillPoly(slice_img, [poly.astype(np.int32)], color=1)

            mask[z_idx] |= slice_img.astype(bool)

        return mask


class StructureSet:
    """Container for RT Structure Set (RTSTRUCT) and its ROIs.

    Use :meth:`StructureSet.from_file` to load a DICOM RT Structure Set. The
    object exposes the discovered ROIs via the ``rois`` list and provides
    helper lookup methods.
    """

    def __init__(self, rois: Optional[List[ROI]] = None):
        self.rois: List[ROI] = rois or []

    @classmethod
    def from_file(cls, ds: PydicomDataset, convert_to_cm: bool = True) -> "StructureSet":
        """Read an RT Structure Set DICOM file and return a StructureSet.

        Parameters
        ----------
        ds
            PydicomDataset representing the RTSTRUCT DICOM file.
        convert_to_cm
            If True, coordinates are converted from mm to cm.
        """
        # Collect ROI definitions (StructureSetROISequence)
        roi_defs = getattr(ds, "StructureSetROISequence", None) or []
        # ROI contour data
        roi_contours = getattr(ds, "ROIContourSequence", None) or []
        # Observations sequence that may contain ROI types / overrides
        roi_obs = getattr(ds, "RTROIObservationsSequence", None) or []
        # Build a mapping from ROINumber -> contour item
        contour_map: Dict[int, PydicomDataset] = {}
        for item in roi_contours:
            num = getattr(item, "ReferencedROINumber", None)
            if num is not None:
                contour_map[int(num)] = item

        # Build a mapping from ROINumber -> observation item
        obs_map: Dict[int, PydicomDataset] = {}
        for item in roi_obs:
            num = getattr(item, "ReferencedROINumber", None)
            if num is not None:
                obs_map[int(num)] = item
                obs_map[int(num)] = item

        rois: List[ROI] = []
        for roi_item in roi_defs:
            roi_number = int(getattr(roi_item, "ROINumber"))
            contour_item = contour_map.get(roi_number)
            obs_item = obs_map.get(roi_number)

            # Create ROI from sequences
            roi = ROI.from_dicom_items(
                roi_item, contour_item, obs_item, convert_to_cm=convert_to_cm
            )

            rois.append(roi)

        return cls(rois=rois)

    def get_roi_by_name(self, name: str) -> Optional[ROI]:
        """Return the first ROI with matching name (case-insensitive)."""
        for r in self.rois:
            if r.name.lower() == name.lower():
                return r
        return None

    def get_roi_by_number(self, number: int) -> Optional[ROI]:
        for r in self.rois:
            if r.roi_number == number:
                return r
        return None

    def get_roi_by_type(self, roi_type: str) -> List[ROI]:
        """Return list of ROIs with matching interpreted type (case-insensitive)."""
        return [r for r in self.rois if r.roi_type and r.roi_type.lower() == roi_type.lower()]

    def __repr__(self) -> str:
        return f"<StructureSet rois={len(self.rois)}>"
