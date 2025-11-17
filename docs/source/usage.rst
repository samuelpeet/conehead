Usage and examples
==================

This section presents a short tutorial and a few usage patterns.

Tutorial: run a single static beam
---------------------------------

1. Prepare inputs: machine settings, HU LUT and DICOM folder with CT + RTPLAN.
2. Choose grid geometry (corner, resolution, num_voxels). The project
   uses cm internally; ensure you convert mm -> cm if required.
3. Run the CLI or call :func:`conehead.runner.run_plan` from Python.

Programmatic call example
-------------------------

.. code-block:: python

    from conehead.runner import run_plan

    run_plan(
        settings_path="Truebeam_6FFF_M120.toml",
        dicom_dir="/data/prostate_case",
        hu_lut_path="Siemens_Confidence.toml",
        corner=(-26.96, -23.12, -10.20),
        resolution=(0.2, 0.2, 0.2),
        num_voxels=(267, 206, 138),
        output_dir="out",
    )

Tips and gotchas
----------------
- Grid corner convention: the `Grid` constructor expects the most
  negative corner of the most negative voxel. When writing DICOM
  ImagePositionPatient you may need to add half a voxel to convert to
  voxel-centre representations.
- Units: DICOM uses mm in many tags; the project uses cm internally.
- If you see unexpected structure masks, check that RTSTRUCT contours
  and CT voxel spacing are interpreted with the same coordinate origin
  and orientation.
