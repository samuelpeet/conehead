import pytest
import numpy as np
from conehead.source import Source
from conehead.geometry import line_block_plane_collision


class TestGeometry:

    def test_line_block_plane_collision_G0(self):
        
        source = Source("varian_clinac_6MV")
        source.gantry = np.float32(0)
        ray_start = source.position
        ray_direction = np.array([0, 1, 0])
        plane_normal = source.v_y
        point = line_block_plane_collision(ray_start, ray_direction, plane_normal)
        correct = np.array([0, 0, 0])
        np.testing.assert_array_almost_equal(correct, point)

    def test_line_block_plane_collision_G90(self):
        
        source = Source("varian_clinac_6MV")
        source.gantry = np.float32(90)
        ray_start = source.position
        ray_direction = np.array([-1, 0, 0.5])
        plane_normal = source.v_y
        point = line_block_plane_collision(ray_start, ray_direction, plane_normal)
        correct = np.array([0, 0, 50])
        np.testing.assert_array_almost_equal(correct, point)

    def test_line_block_plane_collision_C270(self):
        
        source = Source("varian_clinac_6MV")
        source.gantry = np.float32(0)
        source.collimator = np.float32(270)
        ray_start = source.position
        ray_direction = np.array([1, 1, 1])
        plane_normal = source.v_y
        point = line_block_plane_collision(ray_start, ray_direction, plane_normal)
        correct = np.array([100, 0, 100])
        np.testing.assert_array_almost_equal(correct, point)        

    def test_line_block_plane_collision_G315_C45(self):
        
        source = Source("varian_clinac_6MV")
        source.gantry = np.float32(315)
        source.collimator = np.float32(45)
        ray_start = source.position
        ray_direction = np.array([1, 1, 1])
        plane_normal = source.v_y
        point = line_block_plane_collision(ray_start, ray_direction, plane_normal)
        correct = np.array([0, 0, 1/np.sqrt(2)*100])
        np.testing.assert_array_almost_equal(correct, point)
    
    def test_line_block_plane_collision_parallel(self):
        with pytest.raises(RuntimeError):
            source = Source("varian_clinac_6MV")
            source.gantry = np.float32(0)
            ray_start = source.position
            ray_direction = np.array([1, 0, 0])
            plane_normal = source.v_y
            point = line_block_plane_collision(ray_start, ray_direction, plane_normal)
 

    # def test_line_calc_limit_plane_collision(self):
    #     ray_direction = np.array([0, 0, -1])
    #     plane_point = np.array([0, 0, -20])
    #     point = line_calc_limit_plane_collision(ray_direction, plane_point)
    #     correct = np.array([0, 0, -20])
    #     np.testing.assert_array_almost_equal(correct, point)

    # def test_line_calc_limit_plane_collision_parallel(self):
    #     with pytest.raises(RuntimeError):
    #         ray_direction = np.array([1, 0, 0])
    #         plane_point = np.array([0, 0, -20])
    #         line_calc_limit_plane_collision(ray_direction, plane_point)

    # def test_isocentre_plane_position(self):
    #     position = np.array([10.0, 20.0, 50.0])
    #     position_iso = isocentre_plane_position(position, 100.0)
    #     correct = np.array([20.0, 40.0])
    #     np.testing.assert_array_almost_equal(correct, position_iso)