#%%
import numpy as np
import os; os.environ["NUMBA_ENABLE_CUDASIM"] = "0"; os.environ["NUMBA_CUDA_DEBUGINFO"] = "0";
import numba
from numba import cuda
import conehead.dda_3d
import time
import math
from conehead.source import Source
from conehead.block import Block
from conehead.phantom import SimplePhantom
# from conehead.conehead import Conehead
from conehead.kernel import KernelMono

k = KernelMono(egslst_path="kernels/6.0MeV/6.0MeV.egslst")
# %%
