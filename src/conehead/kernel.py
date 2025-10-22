import numpy as np

class KernelMono:

    def __init__(self, egslst_path=None) -> None:
        if egslst_path is not None:
            self._from_egslst(egslst_path)
        else:
            raise NotImplementedError

    def _from_egslst(self, egslst_path: str):
        with open(egslst_path) as f:
            start = 0
            end = 0
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "KINETIC ENERGY OF THE INCIDENT BEAM:" in line:
                    self.energy = float(line.split()[-2])
                if "angle/deg.  radius/cm" in line:
                    start = i + 2
                if "END OF RUN" in line:
                    end = i - 1

            if start == 0 or end == 0:
                raise ImportError("Kernel info could not be found in .egslst file.")

            self.data_raw = np.array(
                [l.split()[:3] for l in lines[start:end]], dtype=np.float32
            )
            self.angles = np.unique(self.data_raw[:, 0])
            self.radii = np.unique(self.data_raw[:, 1])
            n_cones = len(self.angles)
            n_depths = len(self.radii)
            self.kernel = self.data_raw[:, 2].reshape((n_cones, n_depths), copy=True)

            r_edges = np.insert(self.radii, 0, 0)
            self.radii_centres = 0.5 * (r_edges[1:] + r_edges[:-1])
            self.angles_edges = np.insert(self.angles, 0, 0)
            self.angles_centres = 0.5 * (self.angles_edges[1:] + self.angles_edges[:-1])
            self.omegas = 2 * np.pi * (np.cos(np.radians(self.angles_edges[:-1])) - np.cos(np.radians(self.angles_edges[1:])))

            for i in range(self.kernel.shape[0]):
                dR = (r_edges[1:]**3 - r_edges[:-1]**3) / 3
                dPhi = self.omegas
                self.kernel[i, :] = self.kernel[i, :] / (dR * dPhi[i])

            self.kernel = self.kernel / self.kernel.sum()

