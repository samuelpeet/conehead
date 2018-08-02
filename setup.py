from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
import os

os.environ["CFLAGS"] = "-O0"

setup(
    name="Conehead",
    ext_modules=cythonize('conehead/*.pyx'),  # accepts a glob pattern
    include_dirs=[np.get_include()],
    gdb_debug=True
)
