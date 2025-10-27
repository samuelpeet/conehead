C / C++ API (Doxygen via Breathe)
=================================

The C/C++/CUDA API below is pulled from the Doxygen XML (via Breathe).
If you don't see function-level entries, regenerate the Doxygen XML
(`doxygen docs/Doxyfile`) and ensure Doxygen is configured to treat
`.cu` files as C++ and to expand CUDA qualifiers (see `docs/Doxyfile`).

.. doxygenindex::
   :project: conehead

Selective per-file pages (GPU sources)
-------------------------------------

The following files in `src/gpu/` are included. Breathe will render
their documented functions and types when Doxygen has produced
corresponding `<memberdef>` entries in the XML.

.. doxygenfile:: bindings.cu

.. doxygenfile:: d_eff.cu

.. doxygenfile:: d_geo.cu

.. doxygenfile:: dose.cu

.. doxygenfile:: fluence.cu

.. doxygenfile:: mask.cu

.. doxygenfile:: oad.cu

.. doxygenfile:: terma.cu

Python API
----------

Core Python package API. The `conehead` package and its main submodules
are documented below. Use `:members:` to include functions, classes,
and variables exposed in each module.

.. automodule:: conehead
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: conehead.block
    :members:

.. automodule:: conehead.dosegrid
    :members:

.. automodule:: conehead.kernel
    :members:

.. automodule:: conehead.nist
    :members:

.. automodule:: conehead.phantom
    :members:

.. automodule:: conehead.source
    :members:

Notes
-----

- After editing `docs/Doxyfile`, run `doxygen docs/Doxyfile` to refresh
  `docs/doxygen/xml/`.
- Then run Sphinx to produce HTML: `sphinx-build -b html docs/source docs/build/html`.
- If the GPU functions still appear only as source listings in the
  generated HTML, open `docs/doxygen/xml/<file>.xml` and check whether
  it contains `<memberdef kind="function">` entries; if not, the
  Doxygen preprocessing/macros configuration needs more PREDEFINED
  entries to strip CUDA qualifiers or expand macros that hide
  prototypes.
