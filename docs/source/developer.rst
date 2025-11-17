Developer guide
===============

This section is for contributors and developers who want to build and
extend Conehead.

Project layout
--------------

- ``src/conehead``: Python package with core modules (grid, plan,
  structure, calculate, dicom helpers).
- ``src/gpu``: optional GPU-accelerated C/C++/CUDA sources (documented
  via Doxygen + Breathe).
- ``scripts``: small helper scripts for CLI and utilities (e.g.
  ``compute_plan.py``).
- ``tests``: pytest-based test suite.

Developing
---------

1. Create and activate a virtual environment.
2. Install dev dependencies:

.. code-block:: bash

    pip install -e .[dev]

3. Run tests with pytest:

.. code-block:: bash

    pytest -q

Documentation build
-------------------

To build the HTML docs locally (requires Doxygen if you want the C++
API docs):

.. code-block:: bash

    # Generate Doxygen XML (if you have doxygen installed)
    doxygen docs/Doxyfile

    # Build the Sphinx docs
    sphinx-build -b html docs/source docs/build/html

Breathe + Doxygen
-----------------

The Sphinx config uses Breathe to include Doxygen XML. If you add or
change C/C++ files, update `docs/Doxyfile` and regenerate the XML with
``doxygen docs/Doxyfile``.
