import numpy as np
from conehead.conehead import Source, beam_to_global, global_to_beam


class TestSourceMovements:
    def test_gantry_0(self):
        source = Source()
        source.gantry(0)
        correct_pos = np.array([0, 1000, 0])
        np.testing.assert_array_almost_equal(
            correct_pos, source.position, decimal=5
        )
        correct_rot = np.array([0, 0, np.pi])
        np.testing.assert_array_almost_equal(
            correct_rot, source.rotation, decimal=5
        )

    def test_gantry_45(self):
        source = Source()
        source.gantry(45)
        correct_pos = np.array([np.cos(np.pi/4)*1000, np.cos(np.pi/4)*1000, 0])
        np.testing.assert_array_almost_equal(
            correct_pos, source.position, decimal=5
        )
        correct_rot = np.array([0, 0, np.pi*3/4])
        np.testing.assert_array_almost_equal(
            correct_rot, source.rotation, decimal=5
        )

    def test_gantry_225(self):
        source = Source()
        source.gantry(225)
        correct_pos = np.array(
            [-np.cos(np.pi/4)*1000, -np.cos(np.pi/4)*1000, 0]
        )
        np.testing.assert_array_almost_equal(
            correct_pos, source.position, decimal=5
        )
        correct_rot = np.array([0, 0, np.pi*7/4])
        np.testing.assert_array_almost_equal(
            correct_rot, source.rotation, decimal=5
        )

    def test_gantry_270(self):
        source = Source()
        source.gantry(270)
        correct_pos = np.array([-1000, 0, 0])
        np.testing.assert_array_almost_equal(
            correct_pos, source.position, decimal=5
        )
        correct_rot = np.array([0, 0, np.pi*3/2])
        np.testing.assert_array_almost_equal(
            correct_rot, source.rotation, decimal=5
        )

    def test_collimator_90(self):
        source = Source()
        source.collimator(90)
        correct_rot = np.array([0, np.pi * 3 / 2, np.pi])
        np.testing.assert_array_almost_equal(
            correct_rot, source.rotation, decimal=5
        )

    def test_collimator_270(self):
        source = Source()
        source.collimator(270)
        correct_rot = np.array([0, np.pi / 2, np.pi])
        np.testing.assert_array_almost_equal(
            correct_rot, source.rotation, decimal=5
        )


class TestCoordinateTranformations:
    def test_beam_to_global_G0_C0(self):
        """ Test at G0, C0 """
        source = Source()
        source.gantry(0)
        source.collimator(0)
        beam_coords = np.array([1, 2, 3])
        global_coords = beam_to_global(
            beam_coords, source.position, source.rotation
        )
        correct = np.array([-1, 998, 3])
        np.testing.assert_array_almost_equal(correct, global_coords, decimal=5)

    def test_beam_to_global_G90_C0(self):
        """ Test at G90, C0 """
        source = Source()
        source.gantry(90)
        source.collimator(0)
        beam_coords = np.array([1, 2, 0])
        global_coords = beam_to_global(
            beam_coords, source.position, source.rotation
        )
        correct = np.array([998, 1, 0])
        np.testing.assert_array_almost_equal(correct, global_coords, decimal=5)

    def test_beam_to_global_G270_C0(self):
        """ Test at G270, C0 """
        source = Source()
        source.gantry(270)
        source.collimator(0)
        beam_coords = np.array([1, 2, 3])
        global_coords = beam_to_global(
            beam_coords, source.position, source.rotation
        )
        correct = np.array([-998, -1, 3])
        np.testing.assert_array_almost_equal(correct, global_coords, decimal=5)

    def test_beam_to_global_G0_C90(self):
        """ Test at G0, C90 """
        source = Source()
        source.gantry(0)
        source.collimator(90)
        beam_coords = np.array([1, 2, 3])
        global_coords = beam_to_global(
            beam_coords, source.position, source.rotation
        )
        correct = np.array([3, 998, 1])
        np.testing.assert_array_almost_equal(correct, global_coords, decimal=5)

    def test_beam_to_global_G270_C270(self):
        """ Test at G270, C270 """
        source = Source()
        source.gantry(270)
        source.collimator(270)
        beam_coords = np.array([1, 2, 3])
        global_coords = beam_to_global(
            beam_coords, source.position, source.rotation
        )
        correct = np.array([-998, -3, -1])
        np.testing.assert_array_almost_equal(correct, global_coords, decimal=5)

    def test_global_to_beam_G0_C0(self):
        """ Test at G0, C0 """
        source = Source()
        source.gantry(0)
        source.collimator(0)
        global_coords = np.array([1, 2, 3])
        beam_coords = global_to_beam(
            global_coords, source.position, source.rotation
        )
        correct = np.array([-1, 998, 3])
        np.testing.assert_array_almost_equal(correct, beam_coords, decimal=5)

    def test_global_to_beam_G90_C0(self):
        """ Test at G90, C0 """
        source = Source()
        source.gantry(90)
        source.collimator(0)
        global_coords = np.array([1, 2, 3])
        beam_coords = global_to_beam(
            global_coords, source.position, source.rotation
        )
        correct = np.array([2, 999, 3])
        np.testing.assert_array_almost_equal(correct, beam_coords, decimal=5)

    def test_global_to_beam_G270_C0(self):
        """ Test at G270, C0 """
        source = Source()
        source.gantry(270)
        source.collimator(0)
        global_coords = np.array([1, 2, 3])
        beam_coords = global_to_beam(
            global_coords, source.position, source.rotation
        )
        correct = np.array([-2, 1001, 3])
        np.testing.assert_array_almost_equal(correct, beam_coords, decimal=5)

    def test_global_to_beam_G0_C90(self):
        """ Test at G0, C90 """
        source = Source()
        source.gantry(0)
        source.collimator(90)
        global_coords = np.array([1, 2, 3])
        beam_coords = global_to_beam(
            global_coords, source.position, source.rotation
        )
        correct = np.array([3, 998, 1])
        np.testing.assert_array_almost_equal(correct, beam_coords, decimal=5)

    def test_global_to_beam_G270_C270(self):
        """ Test at G270, C270 """
        source = Source()
        source.gantry(270)
        source.collimator(270)
        global_coords = np.array([1, 2, 3])
        beam_coords = global_to_beam(
            global_coords, source.position, source.rotation
        )
        correct = np.array([-3, 1001, -2])
        np.testing.assert_array_almost_equal(correct, beam_coords, decimal=5)
