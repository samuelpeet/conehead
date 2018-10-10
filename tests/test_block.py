from os.path import dirname, abspath, join
import numpy as np
import pytest
import pydicom
from conehead.block import Block
from scipy.interpolate import RegularGridInterpolator

class TestBlock:
    def test_square_block(self):
        block = Block()
        block.set_square(10)
        correct = np.zeros((400, 400))
        correct[150:250, 150:250] = 1.0
        np.testing.assert_array_almost_equal(correct, block.block_values)

    def test_mlc(self):
        """
         10------ ------1
           -----   -----
           ----     ----
           ---       ---
        A  --         --  B
           --         --
           ---       ---
           ----     ----
           -----   -----
          1------ ------10
        """
        # Read test plan
        filename = join(dirname(abspath(__file__)), "RP.3DCRT.dcm")
        plan = pydicom.dcmread(filename, force=True)

        # Extract info from plan
        for beam in plan.BeamSequence:

            if beam.BeamType != 'STATIC':
                raise NotImplementedError(
                    "Only beams with type 'STATIC' are currently implemented."
                )

            # Get boundaries of MLCs
            for collimator in beam.BeamLimitingDeviceSequence:
                if collimator.RTBeamLimitingDeviceType == 'MLCX':
                    mlc_boundaries = collimator.LeafPositionBoundaries

            # Get jaw and MLC positions
            for collimator in beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence:
                if collimator.RTBeamLimitingDeviceType in ('X', 'ASYMX'):
                    jaw_x_positions = collimator.LeafJawPositions
                if collimator.RTBeamLimitingDeviceType in ('Y', 'ASYMY'):
                    jaw_y_positions = collimator.LeafJawPositions
                elif collimator.RTBeamLimitingDeviceType == 'MLCX':
                    mlc_ends = collimator.LeafJawPositions

        # Convert to numpy arrays
        mlc_boundaries = np.array(mlc_boundaries).astype(np.float)
        mlc_ends = np.array(mlc_ends).astype(np.float)
        jaw_x_positions = np.array(jaw_x_positions).astype(np.float)
        jaw_y_positions = np.array(jaw_y_positions).astype(np.float)

        # Convert to tenths of a millimetre
        mlc_boundaries = np.floor(mlc_boundaries * 10)
        mlc_ends = np.floor(mlc_ends * 10)
        jaw_x_positions = np.floor(jaw_x_positions * 10)
        jaw_y_positions = np.floor(jaw_y_positions * 10)

        # Identify A and B bank ends
        mlc_ends_a = mlc_ends[:int(len(mlc_ends)/2)]
        mlc_ends_b = mlc_ends[int(len(mlc_ends)/2):]

        # Total width of MLC bank
        mlc_width = int(np.abs(mlc_boundaries[0] - mlc_boundaries[-1]))

        # Internal class to manage the creation of leaves
        class Leaf:
            def __init__(self, min_bound, max_bound, end, bank):
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
                    self.area = np.fliplr(np.flipud(self.area))
                else:
                    assert False, \
                    "bank must be \'A\' or \'B\'"

            def _leaf_transmission(self, width, height):
                    area = np.ones((width, height)) #* 0.98
                    area[0, :] = 0.20
                    area[1, :] = 0.50
                    area[2, :] = 0.75
                    area[-1, :] = 0.20
                    area[-2, :] = 0.50
                    area[-3, :] = 0.75
                    area[:, -15:] *= np.linspace(1.0, 0.0, 15)
                    return 1 - area

        # Create leaves
        leaves_a, leaves_b = [], []
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
        block_values = np.ones((mlc_width, 4000))
        for l in leaves_a:
            block_values[l.r_min:l.r_max, l.c_min:l.c_max] = l.area
        for l in leaves_b:
            block_values[l.r_min:l.r_max, l.c_min:l.c_max] = l.area

        # Include jaws in black plane
        block_values[:, :int(2000+jaw_x_positions[0])] = 1.0
        block_values[:, int(2000+jaw_x_positions[1]):] = 1.0
        block_values[:int(mlc_width/2+jaw_y_positions[0]), :] = 1.0
        block_values[int(mlc_width/2+jaw_y_positions[1]):, :] = 1.0
 
        xmin, xmax, xnum = (-2000, 2000, 4000)
        ymin, ymax, ynum = (
            mlc_boundaries[0],
            mlc_boundaries[-1],
            mlc_width
        )
        block_locations = np.mgrid[
            xmin:xmax:xnum*1j,
            ymin:ymax:ynum*1j
        ]
        block_values_interp = RegularGridInterpolator(
            (np.linspace(xmin, xmax, xnum),
             np.linspace(ymin, ymax, ynum)),
            block_values,
            method='nearest',
            bounds_error=False,
            fill_value=0
        )

        import matplotlib.pyplot as plt
        plt.imshow(block_values)
        plt.title('MLC Transmission')
        plt.colorbar()
        plt.show()