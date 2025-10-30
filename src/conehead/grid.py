import numpy as np
import numpy.typing as npt


"""Grid utilities used by dose/terma/mask-like volumes.

This module provides a lightweight `Grid` container that stores the
metadata and the 3D (voxel) array used across the codebase. The
`Grid.values` array uses the shape ordering (nz, ny, nx) so indexing is
consistently performed as ``values[z, y, x]``. The class intentionally
keeps behaviour minimal: it normalises inputs to numpy arrays and
initialises a zero-filled values buffer.

Notes
-----
- Coordinates and sizes are plain numpy arrays with dtype float32 for
  ``corner`` and ``resolution`` and int32 for ``num_voxels``.
- The voxel-centre location for voxel index ``(ix, iy, iz)`` is
  ``corner + (ix+0.5, iy+0.5, iz+0.5) * resolution`` when sampling.
"""


class Grid:
    """Lightweight 3D grid container.

    Parameters
    ----------
    num_voxels : array-like of int
        Number of voxels along each axis in the order ``(nx, ny, nz)``.
    corner : array-like of float
        World-space coordinates of the minimal corner (x, y, z) of the
        grid (i.e. the location of voxel index (0, 0, 0)).
    resolution : array-like of float
        Voxel sizes along each axis (dx, dy, dz) in the same units as
        ``corner``. Resolution should be provided in the order
        ``(dx, dy, dz)`` corresponding to x, y, z axes.

    Attributes
    ----------
    num_voxels : numpy.ndarray[int32]
        Integer array with shape (3,) giving (nx, ny, nz).
    corner : numpy.ndarray[float32]
        Float32 array with shape (3,) giving the world-space corner.
    resolution : numpy.ndarray[float32]
        Float32 array with shape (3,) giving voxel sizes (dx, dy, dz).
    values : numpy.ndarray[float32]
        The voxel buffer, initialised to zeros, with shape ``(nz, ny, nx)``.
        Access with ``values[z, y, x]``.

    Examples
    --------
    >>> g = Grid(num_voxels=[100, 80, 60], corner=[-10, -10, -5], resolution=[0.2, 0.2, 0.2])
    >>> g.values.shape
    (60, 80, 100)
    """

    def __init__(
        self,
        num_voxels: npt.ArrayLike,
        corner: npt.ArrayLike,
        resolution: npt.ArrayLike,
    ):
        # Canonicalise shapes and types used throughout the project
        self.num_voxels: npt.NDArray[np.int32] = np.asarray(num_voxels, dtype=np.int32)
        self.corner: npt.NDArray[np.float32] = np.asarray(corner, dtype=np.float32)
        self.resolution: npt.NDArray[np.float32] = np.asarray(resolution, dtype=np.float32)

        # The internal buffer uses ordering (nz, ny, nx) so that the
        # fastest-changing index corresponds to x when flattened.
        self.values: npt.NDArray[np.float32] = np.zeros(
            (self.num_voxels[2], self.num_voxels[1], self.num_voxels[0]), dtype=np.float32
        )
