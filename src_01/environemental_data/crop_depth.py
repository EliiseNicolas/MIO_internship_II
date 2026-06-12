"""
crop_depth.py
=============
Coupe toutes les variables d'un NetCDF au-delà d'une profondeur maximale.

Usage :
    python crop_depth.py input.nc output.nc
    python crop_depth.py input.nc output.nc --max_depth 1000
    python crop_depth.py input.nc output.nc --max_depth 500 --depth_dim depth
"""

import argparse
from pathlib import Path
import xarray as xr


def crop_depth(path_in, path_out, max_depth=1000.0, depth_dim="depth"):
    ds = xr.open_dataset(path_in, decode_times=False)

    if depth_dim not in ds.dims:
        raise KeyError(
            f"Dimension {depth_dim!r} introuvable. "
            f"Disponibles : {list(ds.dims)}"
        )

    depth = ds[depth_dim].values
    n_before = len(depth)

    ds_crop = ds.where(ds[depth_dim] <= max_depth, drop=True)

    n_after = len(ds_crop[depth_dim].values)
    print(f"Profondeur max   : {max_depth} m")
    print(f"Niveaux avant    : {n_before}  ({depth[0]:.1f} – {depth[-1]:.1f} m)")
    print(f"Niveaux après    : {n_after}   "
          f"({ds_crop[depth_dim].values[0]:.1f} – "
          f"{ds_crop[depth_dim].values[-1]:.1f} m)")
    print(f"Niveaux supprimés: {n_before - n_after}")

    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    ds_crop.to_netcdf(path_out)
    ds.close()

    print(f"\nSauvegardé → {path_out}")
    return path_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coupe un NetCDF au-delà d'une profondeur maximale."
    )
    parser.add_argument("path_in",  help="NetCDF source")
    parser.add_argument("path_out", help="NetCDF de sortie")
    parser.add_argument("--max_depth",  type=float, default=1000.0,
                        help="Profondeur maximale en mètres (défaut: 1000)")
    parser.add_argument("--depth_dim",  default="depth",
                        help="Nom de la dimension profondeur (défaut: depth)")
    args = parser.parse_args()

    crop_depth(args.path_in, args.path_out, args.max_depth, args.depth_dim)