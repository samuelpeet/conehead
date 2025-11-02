import numpy as np
import numpy.typing as npt
from pydicom.dataset import FileDataset


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
        rotation: npt.NDArray[np.float32] = np.array([0, 0, 0], dtype=np.float32),
        plan: FileDataset | None = None,
        settings: dict | None = None,
    ):
        self.rotation = rotation
        if plan and settings:
            if settings["mlc"]["mlc_model"] != "Millennium120":
                raise NotImplementedError("Only Millennium 120 MLC is currently implemented.")
            self._set_from_plan(plan, settings)
        else:
            self.xmin: np.float32 = np.float32(-20)
            self.xmax: np.float32 = np.float32(20)
            self.xnum: np.int32 = np.int32(4000)
            self.xres: np.float32 = self.xnum / (self.xmax - self.xmin)
            self.ymin: np.float32 = np.float32(-20)
            self.ymax: np.float32 = np.float32(20)
            self.ynum: np.int32 = np.int32(4000)
            self.yres: np.float32 = self.ynum / (self.ymax - self.ymin)
            self.block_locations: npt.NDArray[np.float32] = np.mgrid[
                self.xmin : self.xmax : self.xnum * 1j, self.ymin : self.ymax : self.ynum * 1j
            ].astype(np.float32)
            self.block_values: npt.NDArray[np.float32] = np.zeros(
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
        self.block_values.fill(np.float32(0))

        # Set square collimator opening
        x1 = int((self.xnum / 2) - (length / 2) * self.xres)
        x2 = int((self.xnum / 2) + (length / 2) * self.xres)
        y1 = int((self.ynum / 2) - (length / 2) * self.yres)
        y2 = int((self.ynum / 2) + (length / 2) * self.yres)
        self.block_values[x1:x2, y1:y2] = np.float32(1)

        self.x1_jaw_pos = -length / 2
        self.x2_jaw_pos = length / 2
        self.y1_jaw_pos = -length / 2
        self.y2_jaw_pos = length / 2

    def _set_from_plan(self, plan: FileDataset, settings: dict):
        # Extract info from plan
        for beam in plan.BeamSequence:
            if beam.BeamType != "STATIC":
                raise NotImplementedError(
                    "Only beams with type 'STATIC' are currently implemented."
                )

            # Get boundaries of MLCs
            for collimator in beam.BeamLimitingDeviceSequence:
                if collimator.RTBeamLimitingDeviceType == "MLCX":
                    mlc_boundaries: npt.NDArray[np.float32] = np.array(
                        collimator.LeafPositionBoundaries, dtype=np.float32
                    )

            # Get jaw and MLC positions
            for collimator in beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence:
                if collimator.RTBeamLimitingDeviceType in ("X", "ASYMX"):
                    jaw_x_positions: npt.NDArray[np.float32] = np.array(collimator.LeafJawPositions)
                if collimator.RTBeamLimitingDeviceType in ("Y", "ASYMY"):
                    jaw_y_positions: npt.NDArray[np.float32] = np.array(collimator.LeafJawPositions)
                elif collimator.RTBeamLimitingDeviceType == "MLCX":
                    mlc_ends: npt.NDArray[np.float32] = np.array(collimator.LeafJawPositions)

        # Convert to tenths of a millimetre
        mlc_boundaries: npt.NDArray[np.float32] = np.floor(mlc_boundaries * 10)
        mlc_ends: npt.NDArray[np.float32] = np.floor(mlc_ends * 10)
        jaw_x_positions: npt.NDArray[np.float32] = np.floor(jaw_x_positions * 10)
        jaw_y_positions: npt.NDArray[np.float32] = np.floor(jaw_y_positions * 10)

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
        self.block_values = np.ones(
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
            #     self.block_values[x1:x2, y1:y2] *= np.fliplr(
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
            self.block_values[x1:x2, y1:y2] *= np.fliplr(
                leaf.area[x1_offset : (leaf.r_max - leaf.r_min - x2_offset), :]
            )

        # Include jaws in block plane
        x_trans = settings["collimators"]["x_jaw_trans"]
        y_trans = settings["collimators"]["y_jaw_trans"]
        self.block_values[:, : int(2000 + jaw_x_positions[0])] *= x_trans
        self.block_values[:, int(2000 + jaw_x_positions[1]) :] *= x_trans
        self.block_values[: int(2000 / 2 + jaw_y_positions[0]), :] *= y_trans
        self.block_values[int(2000 / 2 + jaw_y_positions[1]) :, :] *= y_trans
