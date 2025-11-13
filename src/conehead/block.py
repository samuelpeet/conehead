"""Block plane and fluence map utilities.

This module defines classes and functions to model the treatment head
collimators (jaws and a Varian Millennium MLC) projected into the
block/leaf plane and to generate low-resolution fluence maps from the
high-resolution block plane values.

Two main classes are provided:

- VarianLeaf: Encapsulates the geometric and transmission properties of a
    single MLC leaf (tongue-and-groove, tip profile and screw recess).
- Block: Represents the entire block/leaf plane for a control point. It
    can be constructed from a `ControlPoint` (RTPLAN-derived) and can
    produce primary and secondary fluence maps after convolution and
    beam-profile corrections.

"""

from typing import Tuple
import numpy as np
import numpy.typing as npt
from conehead.plan import ControlPoint
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator, make_interp_spline


class VarianLeaf:
    """Model a single Varian Millennium MLC leaf.

    The object converts configuration values (from the `settings` dict)
    into internal units (tenths of a millimetre) and exposes a
    transmission profile used when slicing leaves into the global block
    plane.

    Parameters
    ----------
    min_bound : float
        Minimum coordinate of the MLC boundary region (in same units as
        values in `mlc_boundaries` — later converted to tenths of mm).
    min_y, max_y : float
        y-range (row indices) spanned by this leaf within the MLC
        boundary grid.
    end : float
        Leaf end position (in same units as `mlc_positions`) used to set
        the per-leaf screw/tip offsets.
    bank : {'A', 'B'}
        Leaf bank identifier. Bank A is conventionally the left bank,
        B the right; behaviour (indexing and flipping) depends on this
        flag.
    settings : dict
        Global machine settings dictionary. Expects an entry
        ``settings['collimators']['mlc']`` with the MLC geometry and
        transmission values.

    Attributes
    ----------
    area : ndarray
        2D transmission map for this leaf including tongue-and-groove
        modifications (shape depends on the leaf width and configured
        sampling along the leaf length).
    """

    def __init__(
        self,
        min_bound: np.float32,
        min_y: np.float32,
        max_y: np.float32,
        end: np.float32,
        bank: str,
        settings: dict,
    ):
        mlc = settings["collimators"]["mlc"]
        # Convert leaf geometry to tenths of a millimetre
        self.T_meas = np.float32(mlc["mlc_trans"])
        self.z_leaf = np.float32(mlc["mlc_z_leaf"]) * 100
        self.z_screw = np.float32(mlc["mlc_z_screw"]) * 100
        self.x_tip_end = np.float32(mlc["mlc_x_tip_end"]) * 100
        self.x_tip_start = np.float32(mlc["mlc_x_tip_start"]) * 100
        self.x_r = np.float32(mlc["mlc_x_r"]) * 100
        self.x_screw_start = np.float32(mlc["mlc_x_screw_start"]) * 100
        self.y_tg = np.float32(mlc["mlc_y_tg"]) * 100

        self.min_bound = min_y
        self.max_bound = max_y
        self.width = int(np.abs(max_y - min_y))
        self.end = int(end)
        self.bank = bank
        self.r_min = int(min_y + np.abs(min_bound) - self.y_tg)
        self.r_max = int(self.r_min + self.width + 2 * self.y_tg)

        if bank == "A":
            self.c_min = 0
            self.c_max = int(2000 + end)
            self.area = self._leaf_transmission(self.width, self.c_max)

        elif bank == "B":
            self.c_min = int(2000 + end)
            self.c_max = 3999
            self.area = self._leaf_transmission(self.width, self.c_max - self.c_min)
            self.area = np.fliplr(np.flipud(self.area))  # type: ignore

        else:
            assert False, "bank must be 'A' or 'B'"

    def _leaf_transmission(self, width: int, height: int) -> npt.NDArray[np.float32]:
        """Compute the 2D effective leaf thickness and convert to transmission.

        The returned array represents the effective radiological path
        length through the leaf material for each sampled point along
        the leaf and is converted to transmission using an effective
        linear attenuation coefficient derived from the measured bulk
        transmission value.

        Parameters
        ----------
        width : int
            Number of rows (in pixels) spanned by the leaf including the
            tongue-and-groove padding.
        height : int
            Number of sampling points along the leaf length direction.

        Returns
        -------
        transmission : ndarray
            2D array giving transmission values in the range (0, 1].
            dtype is float32.
        """

        x = np.arange(0, height, 1).astype(np.float32)
        z = np.zeros_like(x).astype(np.float32)

        # Calculate leaf thickness profile along the leaf length. The
        # profile uses a rounded tip model for the first zone, a constant
        # thickness through the main leaf body, and a reduced thickness
        # in the screw recess region.
        for i in range(len(x)):
            if x[i] <= self.x_tip_start:
                z[i] = 2.0 * np.sqrt(self.x_r**2 - (self.x_r - x[i]) ** 2)
            elif x[i] < self.x_screw_start:
                z[i] = self.z_leaf
            else:
                z[i] = self.z_leaf - self.z_screw

        # Apply tongue-and-groove partial transmission at the top and
        # bottom rows of the leaf.
        tg_pix = int(width + 2 * self.y_tg)
        area = np.tile(z, (tg_pix, 1))
        tg_num = int(self.y_tg * 2)
        for i in range(tg_num):
            area[i, :] *= 0.5
            area[-(1 + i), :] *= 0.5

        # Convert thickness (area) to transmission using an effective
        # attenuation coefficient. T_meas is the measured bulk
        # transmission for the MLC material and z_leaf is its nominal
        # thickness.
        mu_eff = -np.log(self.T_meas) / self.z_leaf
        transmission = np.exp(-mu_eff * area)
        return transmission


class Block:
    """Represent the collimation/block plane for a control point.

    The Block holds a high-resolution 2D array (`values`) that encodes
    the per-pixel transmission due to jaws and MLC leaves projected into
    the block/leaf plane. Instances may be constructed empty or by
    passing a `ControlPoint` and a `settings` dict — in the latter
    case the MLC geometry for a single control point is sliced into
    the block plane and jaw transmission applied.

    Parameters
    ----------
    settings : dict
        Machine and beam configuration dictionary used when building the
        block from a `ControlPoint` and to configure source/fluence
        generation.
    control_point : conehead.plan.ControlPoint, optional
        If provided, the block plane will be initialised from the
        control point's MLC and jaw positions.
    """

    def __init__(
        self,
        settings: dict,
        control_point: ControlPoint | None = None,
    ):
        self.settings = settings
        if control_point and settings:
            if settings["collimators"]["mlc"]["mlc_model"] != "Millennium120":
                raise NotImplementedError("Only Millennium 120 MLC is currently implemented.")
            self._set_from_control_point(control_point, settings)
        else:
            self.xmin: np.float32 = np.float32(-20)
            self.xmax: np.float32 = np.float32(20)
            self.xnum: np.int32 = np.int32(4000)
            self.xres: np.float32 = self.xnum / (self.xmax - self.xmin)
            self.ymin: np.float32 = np.float32(-20)
            self.ymax: np.float32 = np.float32(20)
            self.ynum: np.int32 = np.int32(4000)
            self.yres: np.float32 = self.ynum / (self.ymax - self.ymin)
            self.values: npt.NDArray[np.float32] = np.zeros(
                (self.xnum, self.ynum), dtype=np.float32
            )

    def set_square(self, length: np.float32):
        """Set a simple square aperture in the block plane.

        This helper is mainly intended for tests and simple QA where a
        centred square opening of a given side length is required.

        Parameters
        ----------
        length : float
            Side length (cm) of the square opening centred on the block.
        """
        # Clear previous aperture
        self.values.fill(np.float32(0))

        # Set square collimator opening
        x1 = int((self.xnum / 2) - (length / 2) * self.xres)
        x2 = int((self.xnum / 2) + (length / 2) * self.xres)
        y1 = int((self.ynum / 2) - (length / 2) * self.yres)
        y2 = int((self.ynum / 2) + (length / 2) * self.yres)
        self.values[x1:x2, y1:y2] = np.float32(1)

        self.x1_jaw_pos = -length / 2
        self.x2_jaw_pos = length / 2
        self.y1_jaw_pos = -length / 2
        self.y2_jaw_pos = length / 2

    def _set_from_control_point(self, control_point: ControlPoint, settings: dict):
        """Initialise the Block from an RTPLAN-derived control point.

        The method converts positions expressed in centimetres in the
        `ControlPoint` into the internal sampling units (tenths of a
        millimetre) used by the MLC geometry tables. It then creates
        VarianLeaf objects for each leaf and slices their transmission
        maps into the high-resolution 2D `values` array. Finally jaw
        transmissions and wedge metadata are applied.

        Parameters
        ----------
        control_point : conehead.plan.ControlPoint
            Parsed control point containing `mlc_boundaries`,
            `mlc_positions` and `jaw_*` positions (all in cm).
        settings : dict
            Machine settings dictionary (expects MLC geometry and jack
            jaw transmission entries).
        """

        # Convert from cm to tenths of a mm
        mlc_boundaries = control_point.mlc_boundaries * 100
        mlc_ends = np.floor(control_point.mlc_positions * 100)
        jaw_x_positions = np.floor(control_point.jaw_x_positions * 100)
        jaw_y_positions = np.floor(control_point.jaw_y_positions * 100)

        # Identify A and B bank ends
        mlc_offset = np.float32(
            settings["collimators"]["mlc"]["mlc_offset"] * 100
        )  # cm to tenths of mm
        mlc_ends_a = mlc_ends[: int(len(mlc_ends) / 2)] - mlc_offset
        mlc_ends_b = mlc_ends[int(len(mlc_ends) / 2) :] + mlc_offset

        # Create leaves
        leaves = []
        for n in range(len(mlc_boundaries) - 1):
            leaf = VarianLeaf(
                mlc_boundaries[0],
                mlc_boundaries[n],
                mlc_boundaries[n + 1],
                mlc_ends_a[n],
                "A",
                settings,
            )
            leaves.append(leaf)
        for n in range(len(mlc_boundaries) - 1):
            leaf = VarianLeaf(
                mlc_boundaries[0],
                mlc_boundaries[n],
                mlc_boundaries[n + 1],
                mlc_ends_b[n],
                "B",
                settings,
            )
            leaves.append(leaf)

        # Slice each MLC leaf into the block plane.
        self.values = np.ones(
            (int(np.abs(mlc_boundaries[0] - mlc_boundaries[-1])), 4000), dtype=np.float32
        )
        for leaf in leaves:
            # Because of tongue-and-groove effect, we have to be careful with slicing the first
            # and last leaf in the bank, otherwise they will spill out of bounds.
            x1_offset = 0
            x2_offset = 0
            if leaf.r_min >= 0:
                x1 = leaf.r_min
            else:
                x1_offset = -leaf.r_min
                x1 = 0
            if leaf.r_max <= 3999:
                x2 = leaf.r_max
            else:
                x2_offset = leaf.r_max - 3999
                x2 = 3999
            y1 = leaf.c_min
            y2 = leaf.c_max
            self.values[x1:x2, y1:y2] *= np.fliplr(
                leaf.area[x1_offset : (leaf.r_max - leaf.r_min - x2_offset), :]
            )

        # Include jaws in block plane
        x_trans = settings["collimators"]["x_jaw_trans"]
        y_trans = settings["collimators"]["y_jaw_trans"]
        self.values[:, : int(2000 + jaw_x_positions[0])] *= x_trans
        self.values[:, int(2000 + jaw_x_positions[1]) :] *= x_trans
        self.values[: int(2000 + jaw_y_positions[0]), :] *= y_trans
        self.values[int(2000 + jaw_y_positions[1]) :, :] *= y_trans

        # Record wedge information
        self.wedge_angle = control_point.wedge_angle
        self.wedge_orientation = control_point.wedge_orientation

    def get_fluence_maps(self) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Generate primary and secondary fluence maps from the block plane.

        The method performs the following high-level steps:

        1. Read source size and fluence scaling parameters from
            ``self.settings``.
        2. Interpolate the high-resolution block plane (`self.values`) to
            a lower-resolution grid representing the fluence plane.
        3. Convolve the interpolated block with Gaussian kernels to
            represent finite source sizes for primary and secondary
            source models.
        4. Apply a radial beam-profile correction (BPC) and optional
            wedge modulation to the primary fluence map.

        Returns
        -------
        fluence_map_pri, fluence_map_sec : ndarray, ndarray
            Primary and secondary fluence maps as float32 arrays with
            the target shape (560, 560) by default.
        """

        # Extract source parameters from settings
        sources = self.settings.get("sources", None)
        if sources is None:
            raise ValueError("Block settings do not contain 'sources' information.")
        pri_s = sources.get("pri_s")
        pri_x = sources.get("pri_x")
        pri_y = sources.get("pri_y")
        sec_s = sources.get("sec_s")
        sec_x = sources.get("sec_x")
        sec_y = sources.get("sec_y")
        if (
            pri_s is None
            or pri_x is None
            or pri_y is None
            or sec_s is None
            or sec_x is None
            or sec_y is None
        ):
            raise ValueError("Block settings 'sources' missing required fields.")
        bpc = self.settings.get("beam_profile_correction")
        if bpc is None:
            raise ValueError("Block settings do not contain 'beam_profile_correction' information.")
        oads = bpc.get("oads")
        factors = bpc.get("factors")
        if oads is None or factors is None:
            raise ValueError("Block settings 'beam_profile_correction' missing required fields.")

        # Define original high res dimensions of the block plane
        x_orig = np.linspace(-20.0, 20.0, 4000)  # 4000 points from -20 to 20
        y_orig = np.linspace(-20.0, 20.0, 4000)  # 4000 points from -20 to 20

        # Define the lower res dimensions of the fluence map to interpolate onto
        x_target = np.linspace(-28.0, 28.0, 560)  # 560 points from -28 to 28
        y_target = np.linspace(-28.0, 28.0, 560)  # 560 points from -28 to 28

        # Perform interpolation from original res to target res
        X_target, Y_target = np.meshgrid(x_target, y_target)
        interpolator = RegularGridInterpolator(
            (x_orig, y_orig), self.values, method="linear", bounds_error=False, fill_value=0
        )
        points_target = np.array([X_target.ravel(), Y_target.ravel()]).T
        block_interpolated = interpolator(points_target)
        block_interpolated_2d = block_interpolated.reshape(X_target.shape)

        # Apply Gaussian filtering to simulate source size
        pixel_pitch_cm = 0.1  # cm
        sigma_pix_x = pri_x / pixel_pitch_cm
        sigma_pix_y = pri_y / pixel_pitch_cm
        pri_fluence = gaussian_filter(
            block_interpolated_2d, sigma=(sigma_pix_x, sigma_pix_y), mode="nearest"
        )
        sigma_pix_x = sec_x / pixel_pitch_cm
        sigma_pix_y = sec_y / pixel_pitch_cm
        sec_fluence = gaussian_filter(
            block_interpolated_2d, sigma=(sigma_pix_x, sigma_pix_y), mode="nearest"
        )

        # Determine beam profile correction as a function of radius
        bpc_interp = make_interp_spline(
            oads,
            factors,
            k=1,
        )
        x = np.arange(-28, 28, 0.1, dtype=np.float32)
        y = np.arange(-28, 28, 0.1, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X**2 + Y**2)
        bpc = bpc_interp(r)

        # Calculate final fluence maps
        fluence_map_pri = pri_s * pri_fluence * bpc
        fluence_map_sec = sec_s * sec_fluence

        # Handle wedge modulation if present.
        if self.wedge_angle is not None:
            wedges = self.settings.get("wedges", None)
            if wedges is None:
                raise ValueError("Block settings do not contain 'wedges' information.")
            p = wedges.get("coefficients", [])
            tan_theta = np.tan(self.wedge_angle * np.pi / 180.0)
            # Calculate fluence scaling along the wedge direction
            fluence_scaling = p[0] - p[1] * tan_theta * (
                1 - (p[2] * (y + 0.6)) - np.exp(p[3] * (y + 0.6))
            )
            fluence_scaling_2d = np.tile(
                fluence_scaling, (560, 1)
            )  # Make into 2D array of repeating rows
            # If the wedge orientation is 0 degrees, flip the fluence scaling array left-right
            if self.wedge_orientation == 180:
                fluence_scaling_2d = np.fliplr(fluence_scaling_2d)
            # Now apply the fluence scaling to the primary fluence map
            fluence_map_pri *= fluence_scaling_2d

        return (fluence_map_pri.astype(np.float32), fluence_map_sec.astype(np.float32))
