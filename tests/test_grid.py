# pyright: reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
import numpy as np
import pytest

from conehead.grid import Grid


def test_default_allocation_and_types():
    # num_voxels provided as plain lists; values should be allocated with
    # ordering (nz, ny, nx) and dtypes canonicalised
    g = Grid(num_voxels=[10, 4, 2], corner=[0.0, 1.0, 2.0], resolution=[0.5, 0.5, 1.0])

    # num_voxels were (nx, ny, nz)
    assert tuple(g.num_voxels.tolist()) == (10, 4, 2)
    # values shape is (nz, ny, nx)
    assert g.values.shape == (2, 4, 10)
    assert g.values.dtype == np.float32
    # corner and resolution converted to float32
    assert g.corner.dtype == np.float32
    assert g.resolution.dtype == np.float32


def test_numpy_input_and_shape_property():
    nv = np.array([4, 3, 2], dtype=np.int64)
    corner = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    res = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    g = Grid(num_voxels=nv, corner=corner, resolution=res)
    # num_voxels should be coerced to int32
    assert g.num_voxels.dtype == np.int32
    assert tuple(g.num_voxels.tolist()) == (4, 3, 2)
    # shape property matches values.shape
    assert g.shape == g.values.shape == (2, 3, 4)


def test_provided_values_preserved_and_casted():
    # Create values in float64 and ensure they're cast to float32
    vals = np.full((2, 3, 4), 2.5, dtype=np.float64)
    g = Grid(num_voxels=[4, 3, 2], corner=[0, 0, 0], resolution=[1, 1, 1], values=vals)

    assert g.values.dtype == np.float32
    assert np.allclose(g.values, 2.5)


def test_wrong_shape_raises():
    # Provide values with a mismatched shape and expect a ValueError
    vals = np.zeros((1, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        Grid(num_voxels=[4, 3, 2], corner=[0, 0, 0], resolution=[1, 1, 1], values=vals)


def test_zero_voxels_allowed_and_empty_buffer():
    # zero-sized axes should create an empty buffer but still succeed
    g = Grid(num_voxels=[0, 1, 2], corner=[0, 0, 0], resolution=[1, 1, 1])
    assert g.values.size == 0
    assert g.values.shape == (2, 1, 0)


def test_independent_buffers_between_instances():
    g1 = Grid(num_voxels=[2, 2, 2], corner=[0, 0, 0], resolution=[1, 1, 1])
    g2 = Grid(num_voxels=[2, 2, 2], corner=[0, 0, 0], resolution=[1, 1, 1])

    g1.values[:] = 7.0
    # g2 should remain zeros
    assert not np.allclose(g2.values, 7.0)
