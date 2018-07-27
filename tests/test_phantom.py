from conehead.phantom import SimplePhantom


class TestPhantom:
    def test_phantom_init(self):
        phantom = SimplePhantom()
        assert(phantom is not None)
        assert(phantom.positions is not None)
        assert(phantom.densities is not None)
