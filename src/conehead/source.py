"""Source geometry utilities.

This module provides a small convenience class, :class:`Source`, which
encodes the geometric transform of an external radiation source used in
the simple analytic head model. The class tracks the source-to-axis
distance (SAD), gantry and collimator angles and exposes a local right-
handed basis (v_x, v_y, v_z) together with the current 3D source
position in patient coordinates.

All angles are expressed in degrees and follow the IEC 61217 gantry
convention used in radiotherapy equipment: gantry rotation is about the
isocentre and collimator rotation is applied in the beam's eye view.
"""

import numpy as np
import numpy.typing as npt


class Source:
    """Simple geometric model of the treatment head source.

    The Source class stores the SAD (source-to-axis distance), gantry
    and collimator angles (degrees) and computes the 3D cartesian
    position of the source together with an orthonormal basis that
    describes the beam coordinate system. The class exposes properties
    for these quantities and updates the geometry automatically when
    angles change.

    Parameters
    ----------
    sad : float, optional
        Source-to-axis distance in centimetres. Default is 100 cm.
    isocenter : array_like, shape (3,), optional
        3D coordinates (x, y, z) of the isocenter in cm. The source
        will rotate around this point. Default is [0, 0, 0].

    Attributes
    ----------
    isocenter : ndarray, shape (3,)
        The point in space around which the source rotates (cm).
    position : ndarray, shape (3,)
        3D coordinates of the source in the patient coordinate system
        (cm). The default origin is such that the isocentre lies at
        (0, 0, 0) and the source sits on the negative Y-axis at
        (0, -SAD, 0) before any rotations.
    v_x, v_y, v_z : ndarray, shape (3,)
        Right-handed orthonormal basis vectors (unit length) defining
        the local beam coordinate system after applying collimator and
        gantry rotations. v_z points approximately along the beam axis
        from the source towards the isocentre.
    """

    def __init__(
        self,
        sad: np.float32 = np.float32(100),
        isocenter: npt.ArrayLike = None,
    ):
        # Initialize source to gantry and collimator zero
        self._sad: np.float32 = sad
        self._gantry: np.float32 = np.float32(0)
        self._collimator: np.float32 = np.float32(0)

        # Store the isocenter (point around which source rotates)
        if isocenter is None:
            self._isocenter: npt.NDArray[np.float32] = np.array([0, 0, 0], dtype=np.float32)
        else:
            self._isocenter: npt.NDArray[np.float32] = np.array(isocenter, dtype=np.float32)

        # Default starting position: source on the negative Y axis
        # at distance SAD from isocenter (units: cm)
        self._position: npt.NDArray[np.float32] = np.array(
            [self._isocenter[0], self._isocenter[1] - self._sad, self._isocenter[2]],
            dtype=np.float32,
        )

        # Basis of source local coordinate system (beam-eye view)
        self.v_x: npt.NDArray[np.float32] = np.array([1, 0, 0], dtype=np.float32)
        self.v_y: npt.NDArray[np.float32] = np.array([0, 1, 0], dtype=np.float32)
        self.v_z: npt.NDArray[np.float32] = np.array([0, 0, 1], dtype=np.float32)

    @property
    def position(self) -> npt.NDArray[np.float32]:
        return self._position

    @position.setter
    def position(self, new_postion: npt.NDArray[np.float32]):
        """Set the source position directly.

        Notes
        -----
        This setter is provided for flexibility (tests or bespoke
        transforms). When setting angles via :attr:`gantry` or
        :attr:`collimator` the position will be recomputed
        automatically; manual changes here will not update angles.
        """

        self._position: npt.NDArray[np.float32] = new_postion

    @property
    def sad(self) -> np.float32:
        return self._sad

    @property
    def isocenter(self) -> npt.NDArray[np.float32]:
        return self._isocenter

    @isocenter.setter
    def isocenter(self, new_isocenter: npt.NDArray[np.float32]):
        """Set a new isocenter and update source geometry.

        Parameters
        ----------
        new_isocenter : array_like, shape (3,)
            New isocenter coordinates (x, y, z) in cm.

        Notes
        -----
        Changing the isocenter will recompute the source position
        to maintain the current gantry and collimator angles around
        the new center of rotation.
        """
        self._isocenter = np.array(new_isocenter, dtype=np.float32)
        self._update_geometry()

    @property
    def gantry(self) -> np.float32:
        return self._gantry

    @gantry.setter
    def gantry(self, theta: np.float32):
        """Set the gantry angle of the source.

        Parameters
        ----------
        theta : float
            The gantry angle in degrees. Must be within the range [0, 360).
        """
        assert theta >= 0 and theta < 360, "Invalid gantry angle"
        self._gantry: np.float32 = theta
        self._update_geometry()

    @property
    def collimator(self) -> np.float32:
        return self._collimator

    @collimator.setter
    def collimator(self, theta: np.float32):
        """Set the collimator angle of the source.

        Parameters
        ----------
        theta : float
            The collimator angle in degrees. Will be normalized to [0, 360).
        """
        self._collimator: np.float32 = np.float32(theta % 360)
        self._update_geometry()

    def _update_geometry(self):
        """Recompute source position and beam basis vectors.

        The update follows the IEC 61217 convention: a gantry rotation is
        applied about the isocentre (Z axis in this simplified model) and
        the collimator rotation is applied in the beam-eye view before
        the gantry rotation. The method computes the new cartesian
        position of the source and two successive rotations to build the
        beam basis (collimator then gantry).

        The source rotates around the isocenter at distance SAD.
        """

        # Compute source cartesian position from the gantry angle.
        # First compute position relative to isocenter, then translate.
        theta = self._gantry
        # Convert to IEC-style polar angle (phi) and compute X/Y relative to isocenter
        phi: np.float32 = (np.float32(90) - theta) % np.float32(360)  # IEC 61217
        x_rel: np.float32 = self.sad * np.cos(phi * np.pi / 180)
        y_rel: np.float32 = self.sad * -np.sin(phi * np.pi / 180)
        z_rel: np.float32 = np.float32(0)  # Source stays in isocenter Z-plane

        # Translate to absolute coordinates by adding isocenter offset
        x: np.float32 = x_rel + self._isocenter[0]
        y: np.float32 = y_rel + self._isocenter[1]
        z: np.float32 = z_rel + self._isocenter[2]
        self.position = np.array([x, y, z], dtype=np.float32)

        # Start from canonical basis and apply collimator then gantry
        # rotations. Collimator rotation is a rotation about the local
        # Y axis; gantry rotation is a rotation about the global Z axis.
        v_x: npt.NDArray[np.float32] = np.array([1, 0, 0], dtype=np.float32)
        v_y: npt.NDArray[np.float32] = np.array([0, 1, 0], dtype=np.float32)
        v_z: npt.NDArray[np.float32] = np.array([0, 0, 1], dtype=np.float32)

        # Collimator rotation (negated to match machine convention)
        t = -self._collimator * np.pi / 180
        r_y = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        new_v_x = np.matmul(r_y, v_x)
        new_v_z = np.matmul(r_y, v_z)

        # Gantry rotation about the Z axis
        p = self._gantry * np.pi / 180
        r_z = np.array([[np.cos(p), -np.sin(p), 0], [np.sin(p), np.cos(p), 0], [0, 0, 1]])
        self.v_x = np.matmul(r_z, new_v_x, dtype=np.float32)
        self.v_y = np.matmul(r_z, v_y, dtype=np.float32)
        self.v_z = np.matmul(r_z, new_v_z, dtype=np.float32)
