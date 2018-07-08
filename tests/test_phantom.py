from conehead.phantom import Phantom


class TestPhantom:
    def test_phantom_init(self):
        phantom = Phantom()
        assert(phantom is not None)
        assert(phantom.positions is not None)
        assert(phantom.densities is not None)
