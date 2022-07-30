import numpy as np
import numpy.typing as npt
from pydicom.dataset import FileDataset
from scipy.interpolate import RegularGridInterpolator  # type: ignore


class Block:

    def __init__(self, rotation: npt.NDArray[np.float32] = np.array([0, 0, 0], dtype=np.float32), plan: FileDataset|None = None):
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
                    self.xmin:self.xmax:self.xnum*1j,
                    self.ymin:self.ymax:self.ynum*1j
                ].astype(np.float32)
            self.block_values: npt.NDArray[np.float32] = np.zeros((self.xnum, self.ynum), dtype=np.float32)
            self.block_values_interp = RegularGridInterpolator(
                (np.linspace(self.xmin, self.xmax, self.xnum),
                np.linspace(self.ymin, self.ymax, self.ynum)),
                self.block_values,
                method='nearest',
                bounds_error=False,
                fill_value=0
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

        transmission = self.block_values[
            int(position[0])-1,
            int(position[1])-1
        ]
        return transmission

    def set_square(self, length: np.float32):
        """ Set the block to have a square opening with a given side length.

        Parameters
        ----------
        length : float
            Side length of square opening
        """
        # Set square collimator opening
        x1 = int((self.xnum / 2) - (length / 2) * self.xres)
        x2 = int((self.xnum / 2) + (length / 2) * self.xres)
        y1 = int((self.ynum / 2) - (length / 2) * self.yres)
        y2 = int((self.ynum / 2) + (length / 2) * self.yres)
        self.block_values[x1:x2, y1:y2] = np.float32(1)
        self.block_values_interp = RegularGridInterpolator(
            (np.linspace(self.xmin, self.xmax, self.xnum),
             np.linspace(self.ymin, self.ymax, self.ynum)),
            self.block_values,
            method='nearest',
            bounds_error=False,
            fill_value=0
        )

    def _set_from_plan(self, plan: FileDataset):
        # Extract info from plan
        for beam in plan.BeamSequence:

            if beam.BeamType != 'STATIC':
                raise NotImplementedError(
                    "Only beams with type 'STATIC' are currently implemented."
                )

            # Get boundaries of MLCs
            for collimator in beam.BeamLimitingDeviceSequence:
                if collimator.RTBeamLimitingDeviceType == 'MLCX':
                    mlc_boundaries: npt.NDArray[np.float32] = np.array(collimator.LeafPositionBoundaries, dtype=np.float32)

            # Get jaw and MLC positions
            for collimator in beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence:
                if collimator.RTBeamLimitingDeviceType in ('X', 'ASYMX'):
                    jaw_x_positions: npt.NDArray[np.float32] = np.array(collimator.LeafJawPositions)
                if collimator.RTBeamLimitingDeviceType in ('Y', 'ASYMY'):
                    jaw_y_positions: npt.NDArray[np.float32] = np.array(collimator.LeafJawPositions)
                elif collimator.RTBeamLimitingDeviceType == 'MLCX':
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
        mlc_ends_a = mlc_ends[:int(len(mlc_ends)/2)]
        mlc_ends_b = mlc_ends[int(len(mlc_ends)/2):]

        # Total width of MLC bank
        mlc_width = int(np.abs(mlc_boundaries[0] - mlc_boundaries[-1]))

        # Internal class to manage the creation of leaves
        class Leaf:
            def __init__(self, min_bound: np.float32, max_bound: np.float32, end: np.float32, bank: str):
                self.min_bound = min_bound
                self.max_bound = max_bound
                self.width = int(np.abs(max_bound - min_bound))
                self.end = int(end)
                self.bank = bank
                self.r_min = int(min_bound + np.abs(mlc_boundaries[0]))
                self.r_max = int(self.r_min + self.width)

                if bank == 'A':
                    self.c_min = 0
                    self.c_max = int(2000 + end)
                    self.area = self._leaf_transmission(self.width, self.c_max)

                elif bank == 'B':
                    self.c_min = int(2000 + end)
                    self.c_max = 3999
                    self.area = self._leaf_transmission(self.width, self.c_max - self.c_min)
                    self.area = np.fliplr(np.flipud(self.area))  #type: ignore
                    
                else:
                    assert False, \
                    "bank must be \'A\' or \'B\'"

            def _leaf_transmission(self, width: int, height: int) -> npt.NDArray[np.float32]:
                    area: npt.NDArray[np.float32] = np.ones((width, height), dtype=np.float32) #* 0.98
                    area[0, :] = 0.20
                    area[1, :] = 0.50
                    area[2, :] = 0.75
                    area[-1, :] = 0.20
                    area[-2, :] = 0.50
                    area[-3, :] = 0.75
                    area[:, -15:] *= np.linspace(1.0, 0.0, 15)
                    return 1 - area

        # Create leaves
        leaves_a: list[Leaf] = []
        leaves_b: list[Leaf] = []
        for n in range(len(mlc_boundaries) - 1):
            leaves_a.append(Leaf(
                    mlc_boundaries[n],
                    mlc_boundaries[n+1],
                    mlc_ends_a[n],
                    'A'
                )
            )
            leaves_b.append(Leaf(
                    mlc_boundaries[n],
                    mlc_boundaries[n+1],
                    mlc_ends_b[n],
                    'B'
                )
            )

        # Slice each MLC leaf into the block plane
        self.block_values = np.ones((mlc_width, 4000), dtype=np.float32)
        for l in leaves_a:
            self.block_values[l.r_min:l.r_max, l.c_min:l.c_max] = l.area
        for l in leaves_b:
            self.block_values[l.r_min:l.r_max, l.c_min:l.c_max] = l.area

        # Include jaws in block plane
        self.block_values[:, :int(2000+jaw_x_positions[0])] = 0.0
        self.block_values[:, int(2000+jaw_x_positions[1]):] = 0.0
        self.block_values[:int(mlc_width/2+jaw_y_positions[0]), :] = 0.0
        self.block_values[int(mlc_width/2+jaw_y_positions[1]):, :] = 0.0

        xmin, xmax, xnum = (-20, 20, 4000)
        ymin, ymax, ynum = (
            mlc_boundaries[0] / 100,
            mlc_boundaries[-1] / 100,
            mlc_width
        )
        self.block_locations = np.mgrid[
            xmin:xmax:xnum*1j,
            ymin:ymax:ynum*1j
        ]
        self.block_values_interp = RegularGridInterpolator(
            (np.linspace(xmin, xmax, xnum),
            np.linspace(ymin, ymax, ynum)),
            self.block_values,
            method='nearest',
            bounds_error=False,
            fill_value=0
        )

        # import matplotlib.pyplot as plt
        # plt.imshow(self.block_values)
        # plt.title('MLC Transmission')
        # plt.colorbar()
        # plt.show()
