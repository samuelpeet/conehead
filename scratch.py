import numpy as np
import matplotlib.pyplot as plt
from conehead.varian_clinac_6MV import weights

E = np.linspace(0.001, 6, 500)
w = weights(E)

out = np.dstack((E, w))

print(out)
