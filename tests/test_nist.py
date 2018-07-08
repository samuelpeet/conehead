import numpy as np
from conehead.nist import mu_Al, mu_W, mu_water


class TestNist:
    def test_mu_Al(self):
        E = 2.0
        correct = 4.324E-02
        np.testing.assert_equal(correct, mu_Al(E))

    def test_mu_W(self):
        E = 2.0
        correct = 4.433E-02
        np.testing.assert_equal(correct, mu_W(E))

    def test_mu_water(self):
        E = 2.0
        correct = 4.942E-02
        np.testing.assert_equal(correct, mu_water(E))
