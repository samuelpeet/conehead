from conehead.phantom import SimplePhantom


class TestPhantom:
    def test_phantom_init(self):
        phantom = SimplePhantom()
        assert(phantom is not None)
        assert(phantom.size is not None)
        assert(phantom.origin is not None)
        assert(phantom.spacing is not None)
        assert(phantom.densities is not None)
