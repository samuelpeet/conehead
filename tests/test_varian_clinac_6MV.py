import numpy as np
from conehead.varian_clinac_6MV import weights


class TestVarianClinac6MV:
    def test_weights(self):
        E = 2.0
        correct = np.array(0.427054)
        np.testing.assert_array_almost_equal(correct, weights(E))
