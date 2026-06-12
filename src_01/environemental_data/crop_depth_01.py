"""
crop_depth.py
=============
Pour un NetCDF grillé (time, depth, latitude, longitude) :
  1. Supprime les niveaux de profondeur > max_depth.
  2. Masque (NaN) les colonnes dont la bathymétrie CMEMS < max_depth.

Usage :
    python crop_depth.py input.nc output.nc --bathy bathy.nc
    python crop_depth.py input.nc output.nc --bathy bathy.nc --max_depth 500
    python crop_depth.py input.nc output.nc --bathy bathy.nc \\
        --max_depth 400 --depth_dim depth --bathy_var deptho
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_bathy(bathy_path: str, bathy_var: str) -> xr.DataArray:
    """Ouvre le fichier CMEMS et retourne la DataArray de bathymétrie 2D."""
    ds_b = xr.open_dataset(bathy_path)
    if bathy_var not in ds_b:
        raise KeyError(
            f"Variable {bathy_var!r} introuvable. "
            f"Disponibles : {list(ds_b.data_vars)}"
        )
    bathy = ds_b[bathy_var].load()
    ds_b.close()
    return np.abs(bathy)   # valeurs négatives sous l'eau → profondeur positive


def interp_bathy_to_grid(bathy: xr.DataArray,
                          lon: np.ndarray,
                          lat: np.ndarray) -> np.ndarray:
    """
    Interpole la bathymétrie sur la grille (lat 1D, lon 1D) du dataset source.
    Retourne un array (n_lat, n_lon).
    """
    lon_dim = next(d for d in bathy.dims if "lon" in d.lower())
    lat_dim = next(d for d in bathy.dims if "lat" in d.lower())

    return bathy.interp(
        {lon_dim: xr.DataArray(lon, dims=lon_dim),
         lat_dim: xr.DataArray(lat, dims=lat_dim)},
        method="nearest",
    ).values.astype(np.float32)   # (n_lat, n_lon)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def crop_depth(path_in:    str,
               path_out:   str,
               bathy_path: str,
               max_depth:  float = 400.0,
               depth_dim:  str   = "depth",
               bathy_var:  str   = "deptho") -> str:

    # ── 1. Ouverture ─────────────────────────────────────────────────────────
    print("=" * 55)
    print("CROP DEPTH — coupure verticale + masque bathymétrique")
    print("=" * 55)

    ds = xr.open_dataset(path_in, decode_times=True)
    print(f"\nSource  : {path_in}")
    print(f"Dims    : {dict(ds.dims)}")

    # Détection des noms de dimensions spatiales
    if depth_dim not in ds.dims:
        raise KeyError(
            f"Dimension profondeur {depth_dim!r} introuvable. "
            f"Disponibles : {list(ds.dims)}"
        )
    lon_dim = next((d for d in ds.dims if "lon" in d.lower()), None)
    lat_dim = next((d for d in ds.dims if "lat" in d.lower()), None)
    if lon_dim is None or lat_dim is None:
        raise KeyError(
            "Dimensions latitude/longitude introuvables. "
            f"Disponibles : {list(ds.dims)}"
        )

    # ── 2. Coupure verticale > max_depth ─────────────────────────────────────
    depth_vals = ds[depth_dim].values
    n_before   = len(depth_vals)

    ds_crop  = ds.where(ds[depth_dim] <= max_depth, drop=True)
    depth_cropped = ds_crop[depth_dim].values
    n_after  = len(depth_cropped)

    print(f"\n[1/3] Coupure verticale")
    print(f"      max_depth         : {max_depth} m")
    print(f"      Niveaux avant     : {n_before}  "
          f"({depth_vals[0]:.1f} – {depth_vals[-1]:.1f} m)")
    print(f"      Niveaux après     : {n_after}   "
          f"({depth_cropped[0]:.1f} – {depth_cropped[-1]:.1f} m)")
    print(f"      Niveaux supprimés : {n_before - n_after}")

    # ── 3. Masque bathymétrique ───────────────────────────────────────────────
    print(f"\n[2/3] Masquage bathymétrique")
    print(f"      Fichier bathy : {bathy_path}")
    print(f"      Variable      : {bathy_var}")

    bathy = load_bathy(bathy_path, bathy_var)

    lon_vals = ds_crop[lon_dim].values   # (n_lon,)
    lat_vals = ds_crop[lat_dim].values   # (n_lat,)

    bathy_grid = interp_bathy_to_grid(bathy, lon_vals, lat_vals)
    # bathy_grid : (n_lat, n_lon)

    shallow_2d = xr.DataArray(
        bathy_grid < max_depth,
        dims=[lat_dim, lon_dim],
        coords={lat_dim: ds_crop[lat_dim], lon_dim: ds_crop[lon_dim]},
    )

    n_total   = shallow_2d.size
    n_shallow = int(shallow_2d.values.sum())
    n_valid   = n_total - n_shallow
    print(f"      Points grille sur fond < {max_depth} m : "
          f"{n_shallow} / {n_total}  ({100*n_shallow/n_total:.1f} %)")
    print(f"      Points conservés                       : "
          f"{n_valid} / {n_total}  ({100*n_valid/n_total:.1f} %)")

    ds_masked = ds_crop.where(~shallow_2d)

    # ── 4. Écriture ───────────────────────────────────────────────────────────
    print(f"\n[3/3] Écriture  →  {path_out}")
    Path(path_out).parent.mkdir(parents=True, exist_ok=True)

    # Encodage de compression pour toutes les variables volumineuses
    encoding = {
        var: {"zlib": True, "complevel": 4}
        for var in ds_masked.data_vars
    }

    ds_masked.to_netcdf(path_out, encoding=encoding)
    ds.close()

    print(f"\nTerminé  →  {path_out}")
    print(f"  Dimensions finales : {dict(ds_masked.dims)}")
    return path_out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Coupe un NetCDF grillé au-delà de max_depth ET masque "
            "les colonnes sur fond bathymétrique < max_depth."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path_in",  help="NetCDF source")
    parser.add_argument("path_out", help="NetCDF de sortie")
    parser.add_argument("--bathy",      required=True,
                        help="Fichier bathymétrie CMEMS (.nc)")
    parser.add_argument("--max_depth",  type=float, default=400.0,
                        help="Seuil bathymétrique et profondeur max (m)")
    parser.add_argument("--depth_dim",  default="depth",
                        help="Nom de la dimension profondeur")
    parser.add_argument("--bathy_var",  default="deptho",
                        help="Nom de la variable bathymétrie dans le fichier CMEMS")
    args = parser.parse_args()

    crop_depth(
        path_in    = args.path_in,
        path_out   = args.path_out,
        bathy_path = args.bathy,
        max_depth  = args.max_depth,
        depth_dim  = args.depth_dim,
        bathy_var  = args.bathy_var,
    )