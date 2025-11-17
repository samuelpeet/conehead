Command-line interface
======================

The project provides a small CLI wrapper around the programmatic API to
run dose calculations from the command line. The script is
``cli.py`` and exposes the following arguments:

.. program:: conehead

.. program:: python cli.py

Options
-------

- ``--settings`` (required)
  Path to the machine settings TOML file.

- ``--hu-lut`` (required)
  Path to the HU lookup table TOML file (maps CT numbers to mass density).

- ``--dicom-dir`` (required)
  Directory containing the CT/RTPLAN/RTSTRUCT DICOM files.

- ``--corner`` (required)
  Grid corner (cm) as ``x,y,z`` (most-negative-voxel-corner convention).

- ``--resolution`` (required)
  Voxel size in cm as ``dx,dy,dz``.

- ``--num-voxels`` (required)
  Number of voxels in each axis as ``nx,ny,nz``.

- ``--output-dir``
  Where to write RTDOSE files (defaults to the DICOM directory).

- ``--no-beam-doses``
  Do not write per-beam RTDOSE files; only write plan-sum.

- ``--info-in-file-name``
  Include plan/beam information in generated filenames.

- ``--no-plan-sum``
  Disable writing the plan-summed RTDOSE file.

Examples
--------

Basic run:

.. code-block:: bash

    python scripts/compute_plan.py --settings settings.toml --hu-lut hulut.toml --dicom-dir /path/to/dicom --corner -26.9,-23.1,-10.2 --resolution 0.2,0.2,0.2 --num-voxels 267,206,138

Dry-run / validation
--------------------

The CLI currently validates inputs and will raise an error for missing
files. A future enhancement is to support a ``--dry-run`` flag which
performs only validation and prints a summary of planned outputs.
