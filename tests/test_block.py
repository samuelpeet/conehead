import numpy as np
from conehead.block import Block
from scipy.interpolate import RegularGridInterpolator  # type: ignore

class TestBlock:
    def test_square_block(self):
        block = Block()
        block.set_square(np.float32(10))
        correct = np.zeros((4000, 4000))
        correct[1500:2500, 1500:2500] = 1.0
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
        assert True