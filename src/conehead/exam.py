"""Utilities for loading CT exams and converting HU -> density.

This module provides the :class:`Exam` helper which can load a DICOM
CT series, apply a user-provided HU->density LUT (TOML file), and
provide interpolated density volumes on other target grids.

All spatial coordinates and resolutions used by the class are stored in
centimetres (cm); incoming DICOM spacing/positions (typically in mm)
are converted accordingly.
"""

import numpy as np
import pydicom
import os
import toml
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from conehead.grid import Grid
from conehead.structure import StructureSet


class Exam:
    """Container and loader for a CT examination.

    The ``Exam`` class encapsulates the following responsibilities:
    - loading an HU->density lookup table (TOML file),
    - loading a DICOM CT series and converting pixel values to mass
        density (g/cc), and
    - resampling the internal density volume onto arbitrary target
        grids.

        Parameters
        ----------
        hu_lut_path : str or None
            Path to a TOML file that contains a ``[LUT]`` table with two
            arrays: ``hu`` (Hounsfield units) and ``density`` (matching
            units, e.g. g/cc). The LUT is used with ``numpy.interp`` to map
            CT values to densities. This parameter is optional unless
            ``dicom_folder`` is provided (see below).
        dicom_dir : str or None, optional
            Path to a folder containing a DICOM CT series. If provided and
            ``densities`` is ``None``, the series will be read and converted
            to a :class:`Grid` stored at ``self.densities``. When a
            DICOM folder is supplied, a valid ``hu_lut_path`` must also be
            supplied so CT Hounsfield units can be mapped to density.
        densities : Grid or None, optional
            If provided, this prebuilt :class:`Grid` (with values in g/cc)
            will be used directly as ``self.densities``. This is useful for
            testing or when constructing synthetic CT phantoms. When both
            ``densities`` and ``dicom_folder`` are provided, ``densities``
            takes precedence and the DICOM folder is ignored.

    Attributes
    ----------
    hu_lut : dict
            Dictionary loaded from the TOML file. Expected to contain keys
            ``"hu"`` and ``"density"`` with sequences of the same length.
    densities : Grid
            A :class:`Grid` instance holding the converted density volume in
            shape ``(nz, ny, nx)``. This attribute is created by
            :meth:`_load_dicom_series` unless a prebuilt ``densities`` Grid
            was supplied to the constructor.

    Notes
    -----
    - The DICOM images are filtered for SOP Class ``CT Image Storage``
        and sorted by the z-component of ``ImagePositionPatient``.
    - The code expects Rescale Slope/Intercept to be present in the
        DICOM header and will apply them to produce Hounsfield units
        before LUT mapping.
    - Spatial units: DICOM positions/spacing are converted from mm to
        cm by multiplying by 0.1 before being stored in the Grid.
    """

    def __init__(
        self,
        hu_lut_path: str | None = None,
        dicom_dir: str | None = None,
        densities: Grid | None = None,
    ):
        """Create an Exam.

        Parameters
        ----------
        hu_lut_path : str
            Path to the HU->density TOML file.
        dicom_folder : str, optional
            Path to a folder containing a DICOM CT series. If provided
            and ``densities`` is None, the series will be loaded and
            converted to a :class:`Grid` stored as ``self.densities``.
        densities : Grid, optional
            If provided, this Grid will be used directly as
            ``self.densities`` (useful for tests or synthetic phantoms).

        Notes
        -----
        If both ``densities`` and ``dicom_folder`` are provided, the
        explicit ``densities`` argument takes precedence and the DICOM
        folder will not be read.
        """
        # Load HU LUT only if provided or required
        self.hu_lut = None
        if hu_lut_path is not None:
            self.hu_lut = self._load_hu_lut(hu_lut_path)

        # If caller supplied a prebuilt Grid, use it directly
        if densities is not None:
            if not isinstance(densities, Grid):
                raise TypeError("densities must be an instance of conehead.grid.Grid")
            self.densities = densities
        # Otherwise load from DICOM if requested
        elif dicom_dir is not None:
            # DICOM loading requires an HU->density LUT
            if self.hu_lut is None:
                raise ValueError("hu_lut_path is required when dicom_dir is provided")
            self._load_dicom_series(dicom_dir)
            self._load_structure_set(dicom_dir)
            self._mask_densities_by_structures()

    def _load_hu_lut(self, hu_lut_path: str) -> dict:
        """Load an HU->density lookup table from a TOML file.

        The TOML file should contain a top-level ``[LUT]`` table with two
        arrays: ``hu`` and ``density``. These arrays are used with
        ``numpy.interp`` to translate Hounsfield units to mass
        density (g/cc).

        Parameters
        ----------
        hu_lut_path : str
            Path to the TOML file.

        Returns
        -------
        dict
            The parsed LUT dictionary (``settings['LUT']``).
        """
        settings = toml.load(hu_lut_path)
        if "LUT" not in settings:
            raise ValueError(
                f"HU LUT file {hu_lut_path!r} does not contain a top-level [LUT] table"
            )
        lut = settings["LUT"]
        # Basic validation of LUT contents
        if "hu" not in lut or "density" not in lut:
            raise ValueError(f"LUT table in {hu_lut_path!r} must contain 'hu' and 'density' arrays")
        if len(lut["hu"]) != len(lut["density"]):
            raise ValueError(f"HU and density arrays in {hu_lut_path!r} must have the same length")
        return lut

    def _load_dicom_series(self, dicom_folder: str):
        """Load a DICOM CT series and convert to a :class:`Grid` of
        densities.

        The method searches the provided folder for files ending with
        ``.dcm``, reads them with pydicom, filters to CT instances,
        sorts by slice location, constructs a 3D ndarray, applies the
        DICOM Rescale Slope/Intercept to produce Hounsfield units, and
        maps those to density using the loaded LUT.

        The resulting density volume is stored in ``self.densities`` as
        a :class:`Grid` with units in cm.

        Parameters
        ----------
        dicom_folder : str
            Directory containing the DICOM files for the CT series.
        """
        # Read DICOM files in directory
        dicom_files = []
        for f in os.listdir(dicom_folder):
            if not f.endswith(".dcm"):
                continue
            path = os.path.join(dicom_folder, f)
            try:
                ds = pydicom.dcmread(path)
            except Exception:
                # Skip files that fail to parse
                continue
            dicom_files.append(ds)
        # Remove any non-CT images
        dicom_files = [
            f
            for f in dicom_files
            if getattr(f, "SOPClassUID", None) is not None
            and f.SOPClassUID.name == "CT Image Storage"
        ]

        if len(dicom_files) == 0:
            raise ValueError(f"No CT DICOM files found in folder: {dicom_folder}")

        # Sort slices by ImagePositionPatient (z coordinate)
        # Ensure ImagePositionPatient is present and sort slices by z
        for ds in dicom_files:
            if not hasattr(ds, "ImagePositionPatient"):
                raise ValueError(
                    "DICOM slice missing ImagePositionPatient; cannot determine slice ordering"
                )
        dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # Get image dimensions, spacing, and metadata from the first slice
        first_slice = dicom_files[0]

        # Validate required tags on first slice
        required = [
            "StudyInstanceUID",
            "StudyDate",
            "StudyTime",
            "StudyID",
            "Columns",
            "SliceThickness",
            "PixelSpacing",
            "RescaleSlope",
            "RescaleIntercept",
        ]
        for tag in required:
            if not hasattr(first_slice, tag):
                raise ValueError(f"First DICOM slice missing required tag: {tag}")

        self.study_instance_uid = first_slice.StudyInstanceUID
        self.study_date = first_slice.StudyDate
        self.study_time = first_slice.StudyTime
        self.study_id = first_slice.StudyID

        rows = first_slice.Rows
        cols = first_slice.Columns
        slice_thickness = first_slice.SliceThickness
        pixel_spacing = first_slice.PixelSpacing

        # Create an empty 3D array to store all pixel data
        ct_volume = np.zeros((len(dicom_files), rows, cols), dtype=np.float32)

        # Populate the 3D array
        for i, ds in enumerate(dicom_files):
            # ensure pixel_array is accessible
            if not hasattr(ds, "pixel_array"):
                raise ValueError(f"DICOM file {i} has no pixel data")
            ct_volume[i, :, :] = ds.pixel_array

        # Apply Rescale Slope and Intercept
        rescale_slope = first_slice.RescaleSlope
        rescale_intercept = first_slice.RescaleIntercept
        ct_volume = ct_volume * rescale_slope + rescale_intercept

        # Convert HU to density
        lut = self.hu_lut
        if lut is None:
            # This should not happen because callers that request DICOM loading
            # are required to provide an HU LUT at construction time.
            raise RuntimeError("HU LUT not loaded; cannot convert HU to density")

        ct_volume = np.interp(ct_volume, lut["hu"], lut["density"]).astype(np.float32)

        # Construct the Grid object that holds the densities
        num_voxels = np.array((cols, rows, len(dicom_files)), dtype=np.int32)
        corner = np.array(first_slice.ImagePositionPatient, dtype=np.float32)
        corner *= 0.1  # convert from mm to cm
        resolution = np.array(
            [pixel_spacing[0], pixel_spacing[1], slice_thickness], dtype=np.float32
        )  # dx, dy, dz
        resolution *= 0.1  # convert from mm to cm
        self.densities = Grid(
            num_voxels=num_voxels,
            corner=corner,
            resolution=resolution,
            values=ct_volume,
        )

    def _load_structure_set(self, dicom_folder: str):
        """Load structures from a DICOM RT Structure Set in the provided
        folder.

        Parameters
        ----------
        dicom_folder : str
            Directory containing the DICOM files for the CT series
            and RT Structure Set.
        """

        dicom_files = []
        for f in os.listdir(dicom_folder):
            if not f.endswith(".dcm"):
                continue
            path = os.path.join(dicom_folder, f)
            try:
                ds = pydicom.dcmread(path)
            except Exception:
                # Skip files that fail to parse
                print(f"Failed to read DICOM file: {path}")
                continue
            dicom_files.append(ds)
        # Remove any non-RT STRUCT images
        dicom_files = [
            f
            for f in dicom_files
            if getattr(f, "SOPClassUID", None) is not None
            and f.SOPClassUID.name == "RT Structure Set Storage"
        ]
        if len(dicom_files) == 0:
            # No RT Struct file found
            return None
        if len(dicom_files) > 1:
            raise ValueError(
                f"Multiple RT Structure Set DICOM files found in folder: {dicom_folder}"
            )
        self.structure_set = StructureSet.from_file(dicom_files[0])

    def _mask_densities_by_structures(self):
        """Create a masked density grid where voxels outside the external structure are
        flagged and structure overrides are applied.

        Returns
        -------
        Grid
            A new :class:`Grid` instance containing the masked density volume.
        """
        if self.structure_set is None:
            # No structures loaded; cannot mask densities
            return self.densities
        if self.densities is None:
            raise ValueError("No densities Grid loaded; cannot mask densities")

        self.densities_masked = Grid(
            num_voxels=self.densities.num_voxels,
            corner=self.densities.corner,
            resolution=self.densities.resolution,
            values=self.densities.values.copy(),  # type: ignore
        )

        # First, apply overrides for all structures with known densities
        for roi in self.structure_set.rois:
            if roi.density is None:
                continue  # No override density specified
            roi_mask = roi.mask_on_grid(self.densities)
            self.densities_masked.values[roi_mask] = roi.density  # type: ignore

        # Map all voxels not enclosed by the external contour or a support structure to -1
        included_roi = self.structure_set.get_roi_by_type("EXTERNAL")
        if not included_roi:
            # No external-like ROI found; this is considered an error in the
            # masking pipeline.
            raise ValueError("No 'External' structure found in StructureSet")

        # Include any support structures as well
        included_roi += self.structure_set.get_roi_by_type("SUPPORT")

        # Masks produced by ROI.mask_on_grid have shape (nz, ny, nx) so build
        # an inclusion mask with the same shape as the density volume.
        inclusion_mask = np.zeros(self.densities.values.shape, dtype=bool)  # type: ignore
        for roi in included_roi:
            roi_mask = roi.mask_on_grid(self.densities)
            inclusion_mask |= roi_mask

        self.densities_masked.values[~inclusion_mask] = -1.0  # type: ignore

    def plot_slice(
        self, z_index: int | None = None, masked=False, ax=None, cmap: str = "gray"
    ) -> None:
        """Plot a single axial slice of the loaded density volume.

        Parameters
        ----------
        z_index : int or None
            Index of the slice along the z axis (0..nz-1). If ``None``,
            the central slice will be plotted.
        ax : matplotlib.axes.Axes or None
            Optional axes to draw into. If ``None``, a new figure and
            axes will be created.
        cmap : str
            Matplotlib colormap name used for plotting.

        Raises
        ------
        ValueError
            If no density volume is loaded on this Exam instance.
        """
        if getattr(self, "densities", None) is None:
            raise ValueError(
                "No density volume loaded; call _load_dicom_series first or provide a dicom_folder to the constructor"
            )
        if masked:
            vol = getattr(self.densities_masked, "values", None)
            if vol is None:
                raise ValueError("No masked density volume available")
        else:
            vol = getattr(self.densities, "values", None)
            if vol is None:
                raise ValueError("Loaded density Grid contains no values to plot")

        nz = vol.shape[0]
        if z_index is None:
            zi = nz // 2
        else:
            zi = z_index

        # Ensure zi is an int for safe comparisons
        zi = int(zi)
        if not (0 <= zi < nz):
            raise IndexError(f"z_index {zi} out of range [0, {nz})")

        slice_img = vol[zi, :, :]
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            created_fig = True

        im = ax.imshow(slice_img, origin="upper", cmap=cmap)
        ax.set_title(f"Density slice z={zi}")
        ax.set_xlabel("x (voxels)")
        ax.set_ylabel("y (voxels)")
        plt.colorbar(im, ax=ax, label="density (g/cc)")
        if created_fig:
            plt.show()

    def densities_on_grid(self, grid: Grid) -> Grid:
        """Resample the internal density volume onto a target :class:`Grid`.

        Parameters
        ----------
        grid : Grid
            Target grid that defines the desired voxel geometry. The
            returned Grid will have the same ``num_voxels``, ``corner``
            and ``resolution`` as the provided ``grid`` but with values
            computed by trilinear interpolation of ``self.densities``.

        Returns
        -------
        Grid
            A new :class:`Grid` instance containing the interpolated
            density volume (dtype float32, shape ``(nz, ny, nx)``).
        """
        interpolator = RegularGridInterpolator(
            (
                self.densities.corner[2]
                + np.arange(self.densities.num_voxels[2]) * self.densities.resolution[2],
                self.densities.corner[1]
                + np.arange(self.densities.num_voxels[1]) * self.densities.resolution[1],
                self.densities.corner[0]
                + np.arange(self.densities.num_voxels[0]) * self.densities.resolution[0],
            ),
            self.densities.values,
            bounds_error=False,
            fill_value=0.0,
        )

        # Build a 3D meshgrid of points in the target grid to sample
        nx, ny, nz = int(grid.num_voxels[0]), int(grid.num_voxels[1]), int(grid.num_voxels[2])
        xs = grid.corner[0] + (np.arange(nx) + 0.5) * grid.resolution[0]
        ys = grid.corner[1] + (np.arange(ny) + 0.5) * grid.resolution[1]
        zs = grid.corner[2] + (np.arange(nz) + 0.5) * grid.resolution[2]
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
        pts = np.stack((Z, Y, X), axis=-1)

        # Evaluate the CT interpolator at these points
        density_vals = interpolator(pts)  # returns shape (nz, ny, nx)

        # Return a new Grid with the interpolated densities
        return Grid(
            num_voxels=grid.num_voxels,
            corner=grid.corner,
            resolution=grid.resolution,
            values=density_vals.astype(np.float32),
        )
