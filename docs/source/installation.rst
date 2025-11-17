Installation
============

This page describes how to install Conehead for development and for
end-users.

Install from source (development)
---------------------------------

1. Create a virtual environment (recommended):

.. code-block:: bash

    python -m venv env
    source env/bin/activate

2. Install development requirements and package in editable mode:

.. code-block:: bash

    pip install -e .[dev]

This installs the package and development tools (pytest, sphinx, ruff,
pre-commit if declared in `pyproject.toml`).

Install a released version
--------------------------

When a package is published to PyPI you can install it with:

.. code-block:: bash

    pip install conehead

System dependencies
-------------------

Some optional features (GPU acceleration, compiled helpers) require a
C/C++ toolchain or CUDA. These are optional and documented in the
project README. The pure-Python functionality (DICOM parsing, grid
handling, simple QA) works with just the Python dependencies.
