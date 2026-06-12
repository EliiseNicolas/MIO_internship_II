"""
create_mini_dataset.py
Crée un sous-ensemble d'un NetCDF acoustique (format ICES AcMeta).

Usage :
    python create_mini_dataset.py input.nc output_mini.nc
    python create_mini_dataset.py input.nc output_mini.nc --n_channel 3 --n_time 1000 --n_depth 25
"""

import argparse
import numpy as np
import xarray as xr


def create_mini_dataset(path_in, path_out, n_channel=5, n_time=1000, n_depth=25,
                        channel_start=0, time_start=0, depth_start=0,
                        seed=None):
    """
    Crée un mini NetCDF à partir d'un fichier acoustique ICES AcMeta.

    Paramètres
    ----------
    path_in        : fichier source
    path_out       : fichier de sortie
    n_channel      : nombre de canaux (fréquences) à garder   — max 5
    n_time         : nombre de pings (time steps) à garder    — max 230091
    n_depth        : nombre de points de profondeur à garder  — max 50
    channel_start  : indice de départ channel  (None = aléatoire)
    time_start     : indice de départ time     (None = aléatoire)
    depth_start    : indice de départ depth    (None = aléatoire)
    seed           : graine pour reproductibilité
    """
    rng = np.random.default_rng(seed)

    ds = xr.open_dataset(path_in)

    # ── Dimensions disponibles ────────────────────────────────────────────
    dims = dict(ds.dims)
    print("Dimensions source :", {k: v for k, v in dims.items() if not k.startswith("STRING")})

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

    sl_channel = make_slice("channel", n_channel, channel_start)
    sl_time    = make_slice("time",    n_time,    time_start)
    sl_depth   = make_slice("depth",   n_depth,   depth_start)

    # ── Sélection (les dimensions STRING* sont ignorées : pas de slicing) ─
    indexers = {}
    if "channel" in dims: indexers["channel"] = sl_channel
    if "time"    in dims: indexers["time"]    = sl_time
    if "depth"   in dims: indexers["depth"]   = sl_depth

    ds_mini = ds.isel(**indexers)

    # ── Copie des attributs globaux + annotation ──────────────────────────
    ds_mini.attrs = dict(ds.attrs)
    ds_mini.attrs["mini_dataset"]    = "True"
    ds_mini.attrs["mini_n_channel"]  = int(ds_mini.dims.get("channel", 0))
    ds_mini.attrs["mini_n_time"]     = int(ds_mini.dims.get("time",    0))
    ds_mini.attrs["mini_n_depth"]    = int(ds_mini.dims.get("depth",   0))
    ds_mini.attrs["mini_source"]     = str(path_in)

    # Nettoyage des anciens attributs mini_ obsolètes si présents
    for old_key in ["mini_n_freqs", "mini_n_pings"]:
        ds_mini.attrs.pop(old_key, None)

    ds.close()

    # ── Sauvegarde ────────────────────────────────────────────────────────
    encoding = {}
    if "time" in ds_mini:
        encoding["time"] = {"dtype": "float64", "units": "seconds since 1970-01-01"}

    ds_mini.to_netcdf(path_out, encoding=encoding)

    print(f"\nMini dataset sauvegardé  : {path_out}")
    print("Dimensions résultantes   :", {
        k: v for k, v in ds_mini.dims.items() if not k.startswith("STRING")
    })
    print("Canaux (fréquences)      :", ds_mini["channel"].values.tolist())
    print("Plage temporelle         :",
          str(ds_mini["time"].values[0])[:19], "→",
          str(ds_mini["time"].values[-1])[:19])
    print("Profondeurs              :",
          f"{ds_mini['depth'].values[0]:.1f} → {ds_mini['depth'].values[-1]:.1f} m")
    print("Variables conservées     :", list(ds_mini.data_vars))

    size_mb = sum(
        v.nbytes for v in ds_mini.data_vars.values()
        if hasattr(v, "nbytes")
    ) / 1e6
    print(f"Taille estimée           : {size_mb:.1f} MB")

    return ds_mini


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crée un mini NetCDF acoustique ICES pour tests rapides."
    )
    parser.add_argument("path_in",  type=str, help="NetCDF source")
    parser.add_argument("path_out", type=str, help="NetCDF de sortie")
    parser.add_argument("--n_channel",    type=int, default=5,
                        help="Nombre de canaux/fréquences (défaut: 5, max: 5)")
    parser.add_argument("--n_time",       type=int, default=1000,
                        help="Nombre de pings/time steps (défaut: 1000)")
    parser.add_argument("--n_depth",      type=int, default=25,
                        help="Nombre de points de profondeur (défaut: 25, max: 50)")
    parser.add_argument("--channel_start",type=int, default=0,
                        help="Indice channel de départ (défaut: 0)")
    parser.add_argument("--time_start",   type=int, default=None,
                        help="Indice time de départ (défaut: aléatoire)")
    parser.add_argument("--depth_start",  type=int, default=0,
                        help="Indice depth de départ (défaut: 0)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Graine aléatoire (défaut: 42)")

    args = parser.parse_args()

    create_mini_dataset(
        path_in       = args.path_in,
        path_out      = args.path_out,
        n_channel     = args.n_channel,
        n_time        = args.n_time,
        n_depth       = args.n_depth,
        channel_start = args.channel_start,
        time_start    = args.time_start,
        depth_start   = args.depth_start,
        seed          = args.seed,
    )