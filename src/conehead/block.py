from typing import Tuple
import numpy as np
import numpy.typing as npt
from conehead.plan import ControlPoint
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator, make_interp_spline


class VarianLeaf:
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
        x = np.arange(0, height, 1).astype(np.float32)
        z = np.zeros_like(x).astype(np.float32)

        # Calculate leaf thickness profile
        for i in range(len(x)):
            if x[i] <= self.x_tip_start:
                z[i] = 2.0 * np.sqrt(self.x_r**2 - (self.x_r - x[i]) ** 2)
            elif x[i] < self.x_screw_start:
                z[i] = self.z_leaf
            else:
                z[i] = self.z_leaf - self.z_screw

        # Tongue-and-groove effect
        tg_pix = int(width + 2 * self.y_tg)
        area = np.tile(z, (tg_pix, 1))
        tg_num = int(self.y_tg * 2)
        for i in range(tg_num):
            area[i, :] *= 0.5
            area[-(1 + i), :] *= 0.5

        # Convert to transmission
        mu_eff = -np.log(self.T_meas) / self.z_leaf
        transmission = np.exp(-mu_eff * area)
        return transmission


class Block:
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
        """Set the block to have a square opening with a given side length.

        Parameters
        ----------
        length : float
            Side length of square opening
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
        for i, leaf in enumerate(leaves):
            # Because of tongue-and-groove effect, we have to be careful with slicing the first
            # and last leaf in the bank, otherwise they will spill out of bounds.
            # if leaf.r_min
            #     x1_offset = 0
            #     x2_offset = 0
            #     x1 = leaf.r_min
            #     x2 = leaf.r_max
            #     y1 = leaf.c_min
            #     y2 = leaf.c_max
            #     self.values[x1:x2, y1:y2] *= np.fliplr(
            #         leaf.area[x1_offset : (leaf.r_max - leaf.r_min - x2_offset), :]
            #     )
            #     continue

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
        self.wedge_direction = control_point.wedge_direction

    def get_fluence_maps(self) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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
            a = wedges.get("coefficients", [])[0]
            b = wedges.get("coefficients", [])[1]
            c = wedges.get("coefficients", [])[2]
            d = wedges.get("coefficients", [])[3]
            theta = self.wedge_angle * np.pi / 180.0  # Convert to radians
            fluence_scaling = a - b * np.tan(theta) * (
                1 - c * (y + 0.6) - np.exp(d * (y + 0.6))
            )  # Yu et al Med Phys 2002 Eq. 1
            fluence_scaling_2d = np.tile(
                fluence_scaling[:, np.newaxis], (1, 560)
            )  # Make into 2D array of repeating columns
            # If the wedge direction is 180 degrees, flip the fluence scaling array up-down
            if self.wedge_direction == 180.0:
                fluence_scaling_2d = np.flipud(fluence_scaling_2d)
            # Now apply the fluence scaling to the primary fluence map
            fluence_map_pri *= fluence_scaling_2d

        return (fluence_map_pri.astype(np.float32), fluence_map_sec.astype(np.float32))
