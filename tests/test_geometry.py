import pytest
import numpy as np
from conehead.source import Source
from conehead.geometry import (
    beam_to_global, global_to_beam, line_block_plane_collision,
    line_calc_limit_plane_collision, isocentre_plane_position
)


class TestGeometry:
    def test_beam_to_global_G0_C0(self):
        """ Test at G0, C0 """
        source = Source("varian_clinac_6MV")
        source.gantry(0)
        source.collimator(0)
        beam_coords = np.array([.1, .2, .3])
        global_coords = beam_to_global(
            beam_coords, source.position, source.rotation
        )
        correct = np.array([.1, .2, 100.3])
        np.testing.assert_array_almost_equal(correct, global_coords, decimal=5)

    def test_beam_to_global_G90_C0(self):
        """ Test at G90, C0 """
        source = Source("varian_clinac_6MV")
        source.gantry(90)
        source.collimator(0)
        beam_coords = np.array([.1, .2, .3])
        global_coords = beam_to_global(
            beam_coords, source.position, source.rotation
        )
        correct = np.array([100.3, .2, -.1])
        np.testing.assert_array_almost_equal(correct, global_coords, decimal=5)

    def test_beam_to_global_G270_C0(self):
        """ Test at G270, C0 """
        source = Source("varian_clinac_6MV")
        source.gantry(270)
        source.collimator(0)
        beam_coords = np.array([.1, .2, .3])
        global_coords = beam_to_global(
            beam_coords, source.position, source.rotation
        )
        correct = np.array([-100.3, .2, .1])
        np.testing.assert_array_almost_equal(correct, global_coords, decimal=5)

    def test_beam_to_global_G0_C90(self):
        """ Test at G0, C90 """
        source = Source("varian_clinac_6MV")
        source.gantry(0)
        source.collimator(90)
        beam_coords = np.array([.1, .2, .3])
        global_coords = beam_to_global(
            beam_coords, source.position, source.rotation
        )
        correct = np.array([-.2, .1, 100.3])
        np.testing.assert_array_almost_equal(correct, global_coords, decimal=5)

    def test_beam_to_global_G270_C270(self):
        """ Test at G270, C270 """
        source = Source("varian_clinac_6MV")
        source.gantry(270)
        source.collimator(270)
        beam_coords = np.array([.1, .2, .3])
        global_coords = beam_to_global(
            beam_coords, source.position, source.rotation
        )
        correct = np.array([-100.3, -.1, .2])
        np.testing.assert_array_almost_equal(correct, global_coords, decimal=5)

    def test_global_to_beam_G0_C0(self):
        """ Test at G0, C0 """
        source = Source("varian_clinac_6MV")
        source.gantry(0)
        source.collimator(0)
        global_coords = np.array([.1, .2, .3])
        beam_coords = global_to_beam(
            global_coords, source.position, source.rotation
        )
        correct = np.array([.1, .2, -99.7])
        np.testing.assert_array_almost_equal(correct, beam_coords, decimal=5)

    def test_global_to_beam_G90_C0(self):
        """ Test at G90, C0 """
        source = Source("varian_clinac_6MV")
        source.gantry(90)
        source.collimator(0)
        global_coords = np.array([.1, .2, .3])
        beam_coords = global_to_beam(
            global_coords, source.position, source.rotation
        )
        correct = np.array([-.3, .2, -99.9])
        np.testing.assert_array_almost_equal(correct, beam_coords, decimal=5)

    def test_global_to_beam_G270_C0(self):
        """ Test at G270, C0 """
        source = Source("varian_clinac_6MV")
        source.gantry(270)
        source.collimator(0)
        global_coords = np.array([.1, .2, .3])
        beam_coords = global_to_beam(
            global_coords, source.position, source.rotation
        )
        correct = np.array([.3, .2, -100.1])
        np.testing.assert_array_almost_equal(correct, beam_coords, decimal=5)

    def test_global_to_beam_G0_C90(self):
        """ Test at G0, C90 """
        source = Source("varian_clinac_6MV")
        source.gantry(0)
        source.collimator(90)
        global_coords = np.array([.1, .2, .3])
        beam_coords = global_to_beam(
            global_coords, source.position, source.rotation
        )
        correct = np.array([.2, -.1, -99.7])
        np.testing.assert_array_almost_equal(correct, beam_coords, decimal=5)

    def test_global_to_beam_G270_C270(self):
        """ Test at G270, C270 """
        source = Source("varian_clinac_6MV")
        source.gantry(270)
        source.collimator(270)
        global_coords = np.array([.1, .2, .3])
        beam_coords = global_to_beam(
            global_coords, source.position, source.rotation
        )
        correct = np.array([-.2, .3, -100.1])
        np.testing.assert_array_almost_equal(correct, beam_coords, decimal=5)

    def test_line_block_plane_collision(self):
        ray_direction = np.array([0, 0, -1])
        point = line_block_plane_collision(ray_direction)
        correct = np.array([0, 0, -100])
        np.testing.assert_array_almost_equal(correct, point)

    def test_line_block_plane_collision_parallel(self):
        with pytest.raises(RuntimeError):
            ray_direction = np.array([1, 0, 0])
            line_block_plane_collision(ray_direction)

    def test_line_calc_limit_plane_collision(self):
        ray_direction = np.array([0, 0, -1])
        plane_point = np.array([0, 0, -20])
        point = line_calc_limit_plane_collision(ray_direction, plane_point)
        correct = np.array([0, 0, -20])
        np.testing.assert_array_almost_equal(correct, point)

    def test_line_calc_limit_plane_collision_parallel(self):
        with pytest.raises(RuntimeError):
            ray_direction = np.array([1, 0, 0])
            plane_point = np.array([0, 0, -20])
            line_calc_limit_plane_collision(ray_direction, plane_point)

    def test_isocentre_plane_position(self):
        position = np.array([10.0, 20.0, 50.0])
        position_iso = isocentre_plane_position(position, 100.0)
        correct = np.array([20.0, 40.0])
        np.testing.assert_array_almost_equal(correct, position_iso)