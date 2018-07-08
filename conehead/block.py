import numpy as np
from scipy.interpolate import RegularGridInterpolator


class Block:

    def __init__(self, rotation=np.array([0, 0, 0])):
        self.rotation = rotation
        self.xmin, self.xmax, self.xnum = (-20, 20, 400)
        self.xres = self.xnum / (self.xmax - self.xmin)
        self.ymin, self.ymax, self.ynum = (-20, 20, 400)
        self.yres = self.ynum / (self.ymax - self.ymin)
        self.block_locations = np.mgrid[
                self.xmin:self.xmax:self.xnum*1j,
                self.ymin:self.ymax:self.ynum*1j
            ]
        self.block_values = np.zeros((self.xnum, self.ynum))
        self.block_values_interp = RegularGridInterpolator(
            (np.linspace(self.xmin, self.xmax, self.xnum),
             np.linspace(self.ymin, self.ymax, self.ynum)),
            self.block_values,
            method='nearest',
            bounds_error=False,
            fill_value=0
        )

    def set_square(self, length):
        """ Set the block to have a square opening with a given side length.

        Parameters
        ----------
        length : float
            Side length of square opening
        """
        # Set square collimator opening
        x1 = int((self.xnum / 2) - (length / 2) * self.xres)
        x2 = int((self.xnum / 2) + (length / 2) * self.xres)
        y1 = int((self.ynum / 2) - (length / 2) * self.yres)
        y2 = int((self.ynum / 2) + (length / 2) * self.yres)
        self.block_values[x1:x2, y1:y2] = 1.0
        self.block_values_interp = RegularGridInterpolator(
            (np.linspace(self.xmin, self.xmax, self.xnum),
             np.linspace(self.ymin, self.ymax, self.ynum)),
            self.block_values,
            method='nearest',
            bounds_error=False,
            fill_value=0
        )
