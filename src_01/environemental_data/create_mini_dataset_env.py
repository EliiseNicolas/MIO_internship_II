"""
create_mini_dataset.py
Crée un sous-ensemble d'un NetCDF en conservant toutes les variables,
coordonnées et attributs, mais en réduisant les dimensions spatiales/temporelles.

Usage :
    python create_mini_dataset.py input.nc output_mini.nc
    python create_mini_dataset.py input.nc output_mini.nc --n_time 5 --n_lat 20 --n_lon 20
"""

import argparse
import numpy as np
import xarray as xr


def create_mini_dataset(path_in, path_out, n_time=5, n_lat=20, n_lon=20,
                        lat_start=0, lon_start=0, time_start=0,
                        seed=None):
    """
    Crée un mini NetCDF à partir d'un fichier existant.

    Paramètres
    ----------
    path_in     : fichier source
    path_out    : fichier de sortie
    n_time      : nombre de time steps à garder
    n_lat       : nombre de points latitude
    n_lon       : nombre de points longitude
    lat_start   : indice de départ latitude  (None = aléatoire)
    lon_start   : indice de départ longitude (None = aléatoire)
    time_start  : indice de départ temps     (None = aléatoire)
    seed        : graine pour reproductibilité des indices aléatoires
    """
    rng = np.random.default_rng(seed)

    ds = xr.open_dataset(path_in)

    # ── Dimensions disponibles ────────────────────────────────────────────
    dims = dict(ds.dims)
    print("Dimensions source :", dims)

    # ── Calcul des slices ─────────────────────────────────────────────────
    def make_slice(dim_name, n_keep, start):
        total = dims.get(dim_name, 0)
        if total == 0:
            return slice(None)
        n_keep = min(n_keep, total)
        if start is None:
            start = int(rng.integers(0, total - n_keep + 1))
        start = max(0, min(start, total - n_keep))
        return slice(start, start + n_keep)

    sl_time = make_slice("time",      n_time, time_start)
    sl_lat  = make_slice("latitude",  n_lat,  lat_start)
    sl_lon  = make_slice("longitude", n_lon,  lon_start)

    # ── Sélection ─────────────────────────────────────────────────────────
    indexers = {}
    if "time"      in dims: indexers["time"]      = sl_time
    if "latitude"  in dims: indexers["latitude"]  = sl_lat
    if "longitude" in dims: indexers["longitude"] = sl_lon

    ds_mini = ds.isel(**indexers)

    # ── Copie des attributs globaux + annotation ──────────────────────────
    ds_mini.attrs = dict(ds.attrs)
    ds_mini.attrs["mini_dataset"]  = "True"
    ds_mini.attrs["mini_n_time"]   = int(ds_mini.dims.get("time",      0))
    ds_mini.attrs["mini_n_lat"]    = int(ds_mini.dims.get("latitude",  0))
    ds_mini.attrs["mini_n_lon"]    = int(ds_mini.dims.get("longitude", 0))
    ds_mini.attrs["mini_source"]   = str(path_in)

    ds.close()

    # ── Sauvegarde ────────────────────────────────────────────────────────
    ds_mini.to_netcdf(path_out)

    print(f"\nMini dataset sauvegardé : {path_out}")
    print("Dimensions résultantes  :", dict(ds_mini.dims))
    print("Variables conservées    :", list(ds_mini.data_vars))

    size_mb = sum(
        v.nbytes for v in ds_mini.data_vars.values()
        if hasattr(v, "nbytes")
    ) / 1e6
    print(f"Taille estimée          : {size_mb:.1f} MB")

    return ds_mini


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crée un mini NetCDF pour tests rapides."
    )
    parser.add_argument("path_in",  type=str, help="NetCDF source")
    parser.add_argument("path_out", type=str, help="NetCDF de sortie")
    parser.add_argument("--n_time",     type=int, default=5,
                        help="Nombre de time steps (défaut: 5)")
    parser.add_argument("--n_lat",      type=int, default=20,
                        help="Nombre de points latitude (défaut: 20)")
    parser.add_argument("--n_lon",      type=int, default=20,
                        help="Nombre de points longitude (défaut: 20)")
    parser.add_argument("--lat_start",  type=int, default=None,
                        help="Indice lat de départ (défaut: aléatoire)")
    parser.add_argument("--lon_start",  type=int, default=None,
                        help="Indice lon de départ (défaut: aléatoire)")
    parser.add_argument("--time_start", type=int, default=None,
                        help="Indice time de départ (défaut: aléatoire)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Graine aléatoire pour reproductibilité (défaut: 42)")

    args = parser.parse_args()

    create_mini_dataset(
        path_in    = args.path_in,
        path_out   = args.path_out,
        n_time     = args.n_time,
        n_lat      = args.n_lat,
        n_lon      = args.n_lon,
        lat_start  = args.lat_start,
        lon_start  = args.lon_start,
        time_start = args.time_start,
        seed       = args.seed,
    )