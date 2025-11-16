"""CLI wrapper to compute a plan's dose using conehead.runner.run_plan.

Example
-------
python cli.py \
    --settings Truebeam_6FFF_M120.toml \
    --dicom-dir "Prostate 3DCRT" \
    --corner -26.96,-23.12,-10.20 \
    --resolution 0.2,0.2,0.2 \
    --num-voxels 267,206,138 \
    --output-dir out \
    --no-beam-doses
"""

from __future__ import annotations
import argparse
import logging
from conehead.runner import run_plan


def _parse_tuple(s: str, cast_type):
    return tuple(cast_type(x) for x in s.split(","))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="conehead")
    p.add_argument("--settings", required=True, help="Path to machine settings TOML")
    p.add_argument("--dicom-dir", required=True, help="Directory with CT/RTPLAN/RTSTRUCT")
    p.add_argument("--corner", required=True, help="Grid corner in cm as x,y,z")
    p.add_argument("--resolution", required=True, help="Voxel size in cm as dx,dy,dz")
    p.add_argument("--num-voxels", required=True, help="nx,ny,nz")
    p.add_argument(
        "--hu-lut",
        required=True,
        help="Path to HU lookup table TOML mapping CT numbers to mass density",
    )
    p.add_argument("--output-dir", default=None, help="Where to write RTDOSE output")
    p.add_argument(
        "--info-in-file-name", action="store_true", help="Include plan/beam info in file names"
    )
    p.add_argument(
        "--beam-doses",
        dest="save_beam_doses",
        action="store_true",
        help="Save per beam RTDOSE files as well as plan sum",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    corner = _parse_tuple(args.corner, float)
    resolution = _parse_tuple(args.resolution, float)
    num_voxels = _parse_tuple(args.num_voxels, int)

    run_plan(
        settings_path=args.settings,
        dicom_dir=args.dicom_dir,
        corner=corner,
        resolution=resolution,
        num_voxels=num_voxels,
        hu_lut_path=args.hu_lut,
        output_dir=args.output_dir,
        info_in_file_name=args.info_in_file_name,
        beam_doses=args.beam_doses,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
