import numpy as np
import numpy.typing as npt
from pydicom.dataset import FileDataset
from scipy.interpolate import RegularGridInterpolator  # type: ignore


class Block:
    def __init__(
        self,
        rotation: npt.NDArray[np.float32] = np.array([0, 0, 0], dtype=np.float32),
        plan: FileDataset | None = None,
    ):
        self.rotation = rotation
        if plan:
            self._set_from_plan(plan)
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
            self.block_values_interp = RegularGridInterpolator(
                (
                    np.linspace(self.xmin, self.xmax, self.xnum),
                    np.linspace(self.ymin, self.ymax, self.ynum),
                ),
                self.block_values,
                method="nearest",
                bounds_error=False,
                fill_value=0,
            )

    def transmission(self, position: npt.NDArray[np.float32]) -> np.float32:
        """Return the transmission value at the given position

        Parameters
        ----------
        position : ndarray
            Position in the isocentre plane at which to return the
            transmission, in cm

        Returns
        -------
        float
            The block transmission value, from 0.0 to 1.0
        """
        position = np.floor(position * np.float32(100))  # Convert tenth of a mm
        position = position + np.float32(2000)

        # Handle position lying outside the defined blocking area
        for coord in position:
            if coord < 0 or coord > 3999:
                return np.float32(0)

        transmission = self.block_values[int(position[0]) - 1, int(position[1]) - 1]
        return transmission

    def set_square(self, length: np.float32):
        """Set the block to have a square opening with a given side length.

        Parameters
        ----------
        length : float
            Side length of square opening
        """
        # Clear previous aperture
        self.block_values[:, :] = np.float32(0)

        # Set square collimator opening
        x1 = int((self.xnum / 2) - (length / 2) * self.xres)
        x2 = int((self.xnum / 2) + (length / 2) * self.xres)
        y1 = int((self.ynum / 2) - (length / 2) * self.yres)
        y2 = int((self.ynum / 2) + (length / 2) * self.yres)
        self.block_values[x1:x2, y1:y2] = np.float32(1)
        self.block_values_interp = RegularGridInterpolator(
            (
                np.linspace(self.xmin, self.xmax, self.xnum),
                np.linspace(self.ymin, self.ymax, self.ynum),
            ),
            self.block_values,
            method="nearest",
            bounds_error=False,
            fill_value=0,
        )

    def _set_from_plan(self, plan: FileDataset):
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

        # Convert to numpy arrays
        # mlc_boundaries: n = np.array(mlc_boundaries, dtype=np.float32)
        # mlc_ends = np.array(mlc_ends).astype(np.float32)
        # jaw_x_positions = np.array(jaw_x_positions).astype(np.float32)
        # jaw_y_positions = np.array(jaw_y_positions).astype(np.float32)

        # Convert to tenths of a millimetre
        mlc_boundaries: npt.NDArray[np.float32] = np.floor(mlc_boundaries * 10)
        mlc_ends: npt.NDArray[np.float32] = np.floor(mlc_ends * 10)
        jaw_x_positions: npt.NDArray[np.float32] = np.floor(jaw_x_positions * 10)
        jaw_y_positions: npt.NDArray[np.float32] = np.floor(jaw_y_positions * 10)

        # Identify A and B bank ends
        mlc_ends_a = mlc_ends[: int(len(mlc_ends) / 2)]
        mlc_ends_b = mlc_ends[int(len(mlc_ends) / 2) :]

        # Total width of MLC bank
        mlc_width = int(np.abs(mlc_boundaries[0] - mlc_boundaries[-1]))

        # # Internal class to manage the creation of leaves
        class Leaf:
            def __init__(
                self, min_bound: np.float32, max_bound: np.float32, end: np.float32, bank: str
            ):
                self.T_meas = 0.02
                self.z_leaf = np.floor(6.1 * 100)
                self.z_screw = np.floor(0.33 * 100)
                self.x_tip_end = np.floor(0.0 * 100)
                self.x_tip_start = np.floor(0.6 * 100)
                self.x_r = np.floor(8.0 * 100)
                self.x_screw_start = np.floor(1.7 * 100)
                self.y_tg = np.floor(0.04 * 100)

                self.min_bound = min_bound
                self.max_bound = max_bound
                self.width = int(np.abs(max_bound - min_bound))
                self.end = int(end)
                self.bank = bank
                self.r_min = int(min_bound + np.abs(mlc_boundaries[0]) - self.y_tg)
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

        # Create leaves
        leaves = []
        for n in range(len(mlc_boundaries) - 1):
            leaf = Leaf(mlc_boundaries[n], mlc_boundaries[n + 1], mlc_ends_a[n], "A")
            leaves.append(leaf)
        for n in range(len(mlc_boundaries) - 1):
            leaf = Leaf(mlc_boundaries[n], mlc_boundaries[n + 1], mlc_ends_b[n], "B")
            leaves.append(leaf)

        # Slice each MLC leaf into the block plane. Most of this code is to handle boundary conditions
        self.block_values = np.ones(
            (int(np.abs(mlc_boundaries[0] - mlc_boundaries[-1])), 4000), dtype=np.float32
        )
        for i, leaf in enumerate(leaves):
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
        self.block_values[:, : int(2000 + jaw_x_positions[0])] = 0.0
        self.block_values[:, int(2000 + jaw_x_positions[1]) :] = 0.0
        self.block_values[: int(mlc_width / 2 + jaw_y_positions[0]), :] = 0.0
        self.block_values[int(mlc_width / 2 + jaw_y_positions[1]) :, :] = 0.0
