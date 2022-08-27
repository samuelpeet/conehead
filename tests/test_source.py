import pytest
import numpy as np
from conehead.source import *


class TestSource:
    def test_sad(self):
        sad: np.float32 = np.float32(50.0)
        source = Source("varian_clinac_6MV", sad=sad)
        assert(source.sad == 50)

    def test_source_not_implemented_error(self):
        with pytest.raises(NotImplementedError):
            Source("varian_clinac_10MV")

    def test_gantry_0(self):
        source = Source("varian_clinac_6MV")
        print(source)
        source.gantry = np.float32(0)
        correct_pos = np.array([0, -100, 0])
        np.testing.assert_array_almost_equal(
            correct_pos, source.position, decimal=5
        )

        np.testing.assert_array_almost_equal(
            source.v_x, np.array([1, 0, 0]), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_y, np.array([0, 1, 0]), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_z, np.array([0, 0, 1]), decimal=5
        )

    def test_gantry_45(self):
        source = Source("varian_clinac_6MV")
        source.gantry = np.float32(45)
        correct_pos = np.array([np.sin(np.pi/4)*100, -np.cos(np.pi/4)*100, 0])
        np.testing.assert_array_almost_equal(
            correct_pos, source.position, decimal=5
        )

        np.testing.assert_array_almost_equal(
            source.v_x, np.array([1, 1, 0])/np.sqrt(2), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_y, np.array([-1, 1, 0])/np.sqrt(2), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_z, np.array([0, 0, 1]), decimal=5
        )

    def test_gantry_225(self):
        source = Source("varian_clinac_6MV")
        source.gantry = np.float32(225)
        correct_pos = np.array(
            [-np.sin(np.pi/4)*100, np.cos(np.pi/4)*100, 0]
        )
        np.testing.assert_array_almost_equal(
            correct_pos, source.position, decimal=5
        )

        np.testing.assert_array_almost_equal(
            source.v_x, np.array([-1, -1, 0])/np.sqrt(2), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_y, np.array([1, -1, 0])/np.sqrt(2), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_z, np.array([0, 0, 1]), decimal=5
        )

    def test_gantry_270(self):
        source = Source("varian_clinac_6MV")
        source.gantry = np.float32(270)
        correct_pos = np.array([-100, 0, 0])
        np.testing.assert_array_almost_equal(
            correct_pos, source.position, decimal=5
        )

        np.testing.assert_array_almost_equal(
            source.v_x, np.array([0, -1, 0]), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_y, np.array([1, 0, 0]), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_z, np.array([0, 0, 1]), decimal=5
        )        

    def test_collimator_90(self):
        source = Source("varian_clinac_6MV")
        source.collimator = np.float32(90)

        np.testing.assert_array_almost_equal(
            source.v_x, np.array([0, 0, 1]), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_y, np.array([0, 1, 0]), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_z, np.array([-1, 0, 0]), decimal=5
        )   

    def test_collimator_270(self):
        source = Source("varian_clinac_6MV")
        source.collimator = np.float32(270)

        np.testing.assert_array_almost_equal(
            source.v_x, np.array([0, 0, -1]), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_y, np.array([0, 1, 0]), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_z, np.array([1, 0, 0]), decimal=5
        )

    def test_gantry_90_collimator_90(self):
        source = Source("varian_clinac_6MV")
        source.gantry = np.float32(90)
        source.collimator = np.float32(90)
        correct_pos = np.array([100, 0, 0])
        np.testing.assert_array_almost_equal(
            correct_pos, source.position, decimal=5
        )

        np.testing.assert_array_almost_equal(
            source.v_x, np.array([0, 0, 1]), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_y, np.array([-1, 0, 0]), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_z, np.array([0, -1, 0]), decimal=5
        )

    def test_gantry_45_collimator_270(self):
        source = Source("varian_clinac_6MV")
        source.gantry = np.float32(45)
        source.collimator = np.float32(270)
        correct_pos = np.array([np.sin(np.pi/4)*100, -np.cos(np.pi/4)*100, 0])
        np.testing.assert_array_almost_equal(
            correct_pos, source.position, decimal=5
        )

        np.testing.assert_array_almost_equal(
            source.v_x, np.array([0, 0, -1]), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_y, np.array([-1, 1, 0])/np.sqrt(2), decimal=5
        )
        np.testing.assert_array_almost_equal(
            source.v_z, np.array([1, 1, 0])/np.sqrt(2), decimal=5
        )

    def test_position_transforms(self):
        source = Source("varian_clinac_6MV")
        source.gantry = np.float32(90)

        pos = np.array([10, 0, 0])
        np.testing.assert_array_almost_equal(
            np.array([0, -10, 0]), np.matmul(source.transform, pos), decimal=5
        )

        pos = np.array([-5, -5, 0])
        np.testing.assert_array_almost_equal(
            np.array([-5, 5, 0]), np.matmul(source.transform, pos), decimal=5
        )

        pos = np.array([0, 20, -20])
        np.testing.assert_array_almost_equal(
            np.array([20, 0, -20]), np.matmul(source.transform, pos), decimal=5
        )        