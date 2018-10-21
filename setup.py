from distutils.core import setup  # pylint: disable=no-name-in-module,import-error
from distutils.extension import Extension  # pylint: disable=no-name-in-module,import-error
from Cython.Build import cythonize
import numpy as np
import os

os.environ["CFLAGS"] = "-O0"

ext_modules = [
    Extension(
        "conehead",
        ["conehead/*.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[np.get_include()],
    )   
]

setup(
    name="Conehead",
    ext_modules=cythonize(ext_modules), 
    include_dirs=[np.get_include()]
)
