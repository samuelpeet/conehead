import numpy as np
import matplotlib.pyplot as plt
from conehead.clinac_6MV_spectrum import psi_E

E = np.linspace(0.001, 7, 10000)
W = psi_E(E)

f = plt.figure()
ax = plt.gca()
ax.plot(E, W)
plt.show()
