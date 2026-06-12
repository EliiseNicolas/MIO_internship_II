"""
concat_netcdf.py
================
Concatène tous les NetCDF d'un dossier en un seul fichier.
Tous les fichiers doivent avoir la même structure.

Usage :
    python concat_netcdf.py --input_dir data/ --output out/merged.nc --dim time
"""

import argparse
import glob
from pathlib import Path

import xarray as xr


def concat_netcdf(input_dir: str, output: str, dim: str = "time",
                  pattern: str = "*.nc") -> None:

    input_dir = Path(input_dir)
    files     = sorted(input_dir.glob(pattern))
    print(files)
    if not files:
        raise FileNotFoundError(f"Aucun fichier .nc trouvé dans {input_dir}")

    print(f"  {len(files)} fichiers retenus :")
    for f in files:
        print(f"    {f.name}")

    # ── vérification intégrité ────────────────────────────────────────────────
    valid_files = []
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                _ = list(ds.data_vars)
            valid_files.append(f)
        except Exception as e:
            print(f"  [!] Ignoré (corrompu) : {f.name}\n      {e}")

    if not valid_files:
        raise RuntimeError("Aucun fichier valide.")

    # ── lecture légère : coords + metadata seulement ─────────────────────────
    with xr.open_dataset(valid_files[0]) as ds0:
        ref_vars = list(ds0.data_vars)
        ref_dims = {d: s for d, s in ds0.dims.items() if d != dim}

    print(f"\n  Variables : {ref_vars}")
    print(f"  Dims fixes : {ref_dims}")
    print(f"  Concaténation sur '{dim}' — écriture variable par variable...\n")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── construction du dataset de sortie variable par variable ───────────────
    out_arrays = {}

    for var in ref_vars:
        print(f"  → {var} ...", end=" ", flush=True)
        chunks = []
        for f in valid_files:
            with xr.open_dataset(f) as ds:
                arr = ds[var].load()   # charge uniquement cette variable
                chunks.append(arr)
        out_arrays[var] = xr.concat(chunks, dim=dim)
        print(f"shape={out_arrays[var].shape}")

    # ── coordonnées non-concaténées (lat, lon, depth, bspline_k...) ──────────
    with xr.open_dataset(valid_files[0]) as ds0:
        extra_coords = {
            c: ds0[c]
            for c in ds0.coords
            if dim not in ds0[c].dims
        }

    ds_merged = xr.Dataset(out_arrays, attrs={})

    # ── sauvegarde ────────────────────────────────────────────────────────────
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_merged.data_vars}
    ds_merged.to_netcdf(out_path, format="NETCDF4", encoding=encoding)
    print(f"\n✓ Sauvegardé : {out_path}")
    print(ds_merged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concaténation de NetCDF identiques",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True,  help="Dossier contenant les .nc")
    parser.add_argument("--output",    required=True,  help="Chemin du fichier de sortie")
    parser.add_argument("--dim",       default="time", help="Dimension de concaténation")
    args = parser.parse_args()

    concat_netcdf(
        input_dir = args.input_dir,
        output    = args.output,
        dim       = args.dim,
    )