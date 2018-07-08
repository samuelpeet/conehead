import numpy as np
from conehead.varian_clinac_6MV import weights


class TestVarianClinac6MV:
    def test_single_weight(self):
        E = 2.0
        correct = np.array(0.427054)
        np.testing.assert_array_almost_equal(correct, weights(E))

    def test_single_weight_too_high(self):
        E = 10.0
        correct = np.array(0.0)
        np.testing.assert_array_almost_equal(correct, weights(E))

    def test_array_of_weight(self):
        E = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        correct = np.array([5.000119e-05,
                            1.898469e-01,
                            2.797353e-01,
                            2.840396e-01,
                            9.111887e-02])
        np.testing.assert_array_almost_equal(correct, weights(E))
