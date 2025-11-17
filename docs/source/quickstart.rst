Quickstart
==========

This quickstart shows the minimum steps needed to run a dose
calculation using Conehead's command-line wrapper. For a full
explanation of options see :doc:`/cli` and :doc:`/usage`.

Prerequisites
-------------
- Python 3.10+ (the project uses typed dataclasses and modern numpy APIs)
- pydicom, numpy, scipy and other dependencies listed in `pyproject.toml`

Simple example
--------------
Assuming you have:

- a machine settings TOML (e.g. ``Truebeam_6FFF_M120.toml``),
- an HU lookup TOML (e.g. ``Siemens_Confidence.toml``), and
- a DICOM folder containing CT, RTPLAN and RTSTRUCT files,

run the CLI:

.. code-block:: bash

    python scripts/compute_plan.py \
      --settings Truebeam_6FFF_M120.toml \
      --hu-lut Siemens_Confidence.toml \
      --dicom-dir /path/to/dicom_folder \
      --corner -26.96,-23.12,-10.20 \
      --resolution 0.2,0.2,0.2 \
      --num-voxels 267,206,138 \
      --output-dir out

By default the CLI writes RTDOSE DICOM files to ``out/``.

If you need a programmatic interface (for notebooks or tests) import
and call :func:`conehead.runner.run_plan`.
