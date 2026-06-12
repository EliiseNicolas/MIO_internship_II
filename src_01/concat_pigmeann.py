#!/usr/bin/env python3
"""
Concatenate multiple .nc files along the time dimension.
Usage: python concat_nc.py <input_dir> <output_file>
"""

import sys
import glob
import os
import subprocess
import xarray as xr
from pathlib import Path


def check_file_hdf(path):
    """Use h5dump (CLI) to check HDF5 validity — no Python C-extension involved."""
    result = subprocess.run(
        ["h5dump", "-H", path],
        capture_output=True, timeout=10
    )
    return result.returncode == 0


def check_files(files):
    valid, corrupt = [], []
    print("\n🔍 Checking file integrity (via h5dump)...")
    for f in files:
        if check_file_hdf(f):
            valid.append(f)
        else:
            print(f"   ⚠️  CORRUPT: {os.path.basename(f)}")
            corrupt.append(f)
    return valid, corrupt


def concat_netcdf(input_dir: str, output_file: str):
    output_abs = str(Path(output_file).resolve())

    pattern = os.path.join(input_dir, "*.nc")
    files = sorted([
        f for f in glob.glob(pattern)
        if str(Path(f).resolve()) != output_abs
    ])

    if not files:
        print(f"❌ No .nc files found in: {input_dir}")
        sys.exit(1)

    print(f"✅ Found {len(files)} files")

    # Check if h5dump is available
    if subprocess.run(["which", "h5dump"], capture_output=True).returncode == 0:
        valid_files, corrupt_files = check_files(files)
        print(f"\n   ✅ Valid   : {len(valid_files)}")
        print(f"   ❌ Corrupt : {len(corrupt_files)}")
        if corrupt_files:
            print("   Corrupt files:")
            for f in corrupt_files:
                print(f"      {os.path.basename(f)}")
    else:
        print("⚠️  h5dump not found, skipping integrity check (install hdf5-tools)")
        valid_files = files

    if not valid_files:
        print("❌ No valid files to concatenate.")
        sys.exit(1)

    print(f"\n⏳ Opening {len(valid_files)} datasets...")
    ds = xr.open_mfdataset(
        valid_files,
        combine="by_coords",
        parallel=False,
        engine="h5netcdf",     # avoid netCDF4 C-extension crash on Python 3.14
        chunks={"time": 1},
    )

    print(f"\n📦 Combined dataset:")
    print(f"   time steps : {ds.dims['time']}")
    print(f"   lat        : {ds.dims['lat']}")
    print(f"   lon        : {ds.dims['lon']}")
    print(f"   variables  : {len(ds.data_vars)}")

    print(f"\n💾 Writing to {output_file} ...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(
        output_file,
        engine="h5netcdf",
        encoding={
            var: {"compression": "gzip", "compression_opts": 4}
            for var in ds.data_vars
        },
    )

    size_gb = Path(output_file).stat().st_size / 1e9
    print(f"✅ Done! Output size: {size_gb:.2f} GB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python concat_nc.py <input_dir> <output_file>")
        sys.exit(1)

    concat_netcdf(sys.argv[1], sys.argv[2])