from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt


"""Grid utilities used by dose/terma/mask-like volumes.

This module provides a lightweight `Grid` container that stores the
metadata and the 3D (voxel) array used across the codebase. The
`Grid.values` array uses the shape ordering ``(nz, ny, nx)`` so indexing
is consistently performed as ``values[z, y, x]``. The dataclass stores
raw input fields and canonicalises them in ``__post_init__``.

Notes
-----
- Coordinates and sizes are stored as numpy arrays with dtype float32 for
  ``corner`` and ``resolution`` and int32 for ``num_voxels``.
- The voxel-centre location for voxel index ``(ix, iy, iz)`` is
  ``corner + (ix+0.5, iy+0.5, iz+0.5) * resolution`` when sampling.
"""


@dataclass(slots=True)
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
    values : Optional[numpy.ndarray]
        Optional preallocated voxel buffer. If ``None`` a zero-filled
        array with shape ``(nz, ny, nx)`` will be created in
        ``__post_init__``. If provided, it will be converted to
        float32 and validated for shape.

    Attributes
    ----------
    num_voxels : numpy.ndarray[int32]
        Integer array with shape (3,) giving (nx, ny, nz).
    corner : numpy.ndarray[float32]
        Float32 array with shape (3,) giving the world-space corner.
    resolution : numpy.ndarray[float32]
        Float32 array with shape (3,) giving voxel sizes (dx, dy, dz).
    values : numpy.ndarray[float32]
        The voxel buffer, initialised to zeros if not provided, with
        shape ``(nz, ny, nx)``. Access with ``values[z, y, x]``.

    Examples
    --------
    >>> g = Grid(num_voxels=[100, 80, 60], corner=[-10, -10, -5], resolution=[0.2, 0.2, 0.2])
    >>> g.values.shape
    (60, 80, 100)
    """

    num_voxels: npt.ArrayLike
    corner: npt.ArrayLike
    resolution: npt.ArrayLike
    values: Optional[npt.NDArray[np.float32]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Canonicalise inputs
        self.num_voxels = np.asarray(self.num_voxels, dtype=np.int32)
        self.corner = np.asarray(self.corner, dtype=np.float32)
        self.resolution = np.asarray(self.resolution, dtype=np.float32)

        nx, ny, nz = int(self.num_voxels[0]), int(self.num_voxels[1]), int(self.num_voxels[2])

        if self.values is None:
            # Allocate a zero-filled buffer with ordering (nz, ny, nx)
            self.values = np.zeros((nz, ny, nx), dtype=np.float32)
        else:
            arr = np.asarray(self.values, dtype=np.float32)
            if arr.shape != (nz, ny, nx):
                raise ValueError(f"values has wrong shape {arr.shape}; expected {(nz, ny, nx)}")
            self.values = arr

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the grid shape as (nz, ny, nx)."""

        return self.values.shape
