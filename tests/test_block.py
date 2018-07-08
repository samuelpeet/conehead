import numpy as np
from conehead.block import Block


class TestBlock:
    def test_square_block(self):
        block = Block()
        block.set_square(10)
        correct = np.zeros((400, 400))
        correct[150:250, 150:250] = 1.0
        np.testing.assert_array_almost_equal(correct, block.block_values)
