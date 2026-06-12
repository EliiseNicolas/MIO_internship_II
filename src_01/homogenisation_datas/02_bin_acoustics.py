#!/usr/bin/env python3
"""
bin_acoustic.py — Binning temporel et spatial du Sv.

Opérations :
  - Moyenne temporelle : bins de 200 s (NaN si < 10 pings dans le bin)
  - Moyenne spatiale   : bins de 10 m en profondeur (moyenne linéaire sur Sv en dB)

Usage :
    python bin_acoustic.py \\
        --input  /chemin/vers/input.nc \\
        --output /chemin/vers/output.nc \\
        [--time-bin  200]   # secondes (défaut: 200)
        [--depth-bin  10]   # mètres   (défaut: 10)
        [--min-pings  10]   # pings min pour valider un bin (défaut: 10)

Dépendances :
    pip install xarray numpy pandas netcdf4
"""

import argparse
import sys
import time as _time

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def progress(msg: str):
    print(f"[{_time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bin_sv_time(sv, time_unix, lat, lon, day, bin_sec=200, min_pings=10):
    """
    Moyenne de Sv (en linéaire) sur des bins temporels de bin_sec secondes.
    Retourne NaN si le bin contient moins de min_pings pings.

    sv       : (n_freq, n_ping, n_depth)  float32
    time_unix: (n_ping,) float64  secondes Unix
    lat, lon : (n_ping,) float64
    day      : (n_ping,) int8

    Retourne :
        sv_bin   : (n_freq, n_bins, n_depth)
        time_bin : (n_bins,)  milieu du bin en secondes Unix
        lat_bin  : (n_bins,)
        lon_bin  : (n_bins,)
        day_bin  : (n_bins,)
        counts   : (n_bins,)  nombre de pings par bin
    """
    t_min = time_unix[0]
    t_max = time_unix[-1]
    edges = np.arange(t_min, t_max + bin_sec, bin_sec)
    n_bins = len(edges) - 1

    n_freq, n_ping, n_depth = sv.shape

    sv_bin   = np.full((n_freq, n_bins, n_depth), np.nan, dtype=np.float32)
    time_bin = np.full(n_bins, np.nan, dtype=np.float64)
    lat_bin  = np.full(n_bins, np.nan, dtype=np.float64)
    lon_bin  = np.full(n_bins, np.nan, dtype=np.float64)
    day_bin  = np.full(n_bins, np.nan, dtype=np.float64)
    counts   = np.zeros(n_bins, dtype=np.int32)

    # Index de bin pour chaque ping
    bin_idx = np.searchsorted(edges, time_unix, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Sv linéaire pour la moyenne
    sv_lin = 10.0 ** (sv / 10.0)   # (n_freq, n_ping, n_depth)

    for b in range(n_bins):
        mask = bin_idx == b
        n = mask.sum()
        counts[b] = n

        time_bin[b] = edges[b] + bin_sec / 2.0

        if n < min_pings:
            continue  # NaN déjà initialisé

        sv_lin_mean = np.nanmean(sv_lin[:, mask, :], axis=1)   # (n_freq, n_depth)
        with np.errstate(divide='ignore', invalid='ignore'):
            sv_db = 10.0 * np.log10(sv_lin_mean)
            sv_db[sv_lin_mean <= 0] = np.nan
        sv_bin[:, b, :] = sv_db.astype(np.float32)

        lat_bin[b] = np.nanmean(lat[mask])
        lon_bin[b] = np.nanmean(lon[mask])
        day_bin[b] = np.round(np.nanmean(day[mask].astype(float))).astype(int)

    return sv_bin, time_bin, lat_bin, lon_bin, day_bin.astype(np.int8), counts


def bin_sv_depth(sv, depth, bin_m=10):
    """
    Moyenne de Sv (en linéaire) sur des bins de profondeur de bin_m mètres.

    sv    : (n_freq, n_time, n_depth)
    depth : (n_depth,)

    Retourne :
        sv_bin    : (n_freq, n_time, n_depth_bin)
        depth_bin : (n_depth_bin,)  milieu de chaque bin
    """
    d_min = depth[0]
    d_max = depth[-1]
    edges = np.arange(np.floor(d_min / bin_m) * bin_m,
                      d_max + bin_m, bin_m)
    n_bins = len(edges) - 1
    depth_bin = edges[:-1] + bin_m / 2.0

    n_freq, n_time, n_depth = sv.shape
    sv_bin = np.full((n_freq, n_time, n_bins), np.nan, dtype=np.float32)

    bin_idx = np.searchsorted(edges, depth, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    sv_lin = 10.0 ** (sv / 10.0)   # (n_freq, n_time, n_depth)

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        mean_lin = np.nanmean(sv_lin[:, :, mask], axis=2)   # (n_freq, n_time)
        with np.errstate(divide='ignore', invalid='ignore'):
            sv_db = 10.0 * np.log10(mean_lin)
            sv_db[mean_lin <= 0] = np.nan
        sv_bin[:, :, b] = sv_db.astype(np.float32)

    return sv_bin, depth_bin


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Binning temporel (200 s) et spatial (10 m) du Sv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--time-bin",   type=float, default=200.0,
                        help="Durée des bins temporels (secondes)")
    parser.add_argument("--depth-bin",  type=float, default=10.0,
                        help="Épaisseur des bins de profondeur (mètres)")
    parser.add_argument("--min-pings",  type=int,   default=10,
                        help="Nombre minimum de pings pour valider un bin temporel")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Lecture
    # ------------------------------------------------------------------
    progress(f"Lecture de {args.input}")
    ds = xr.open_dataset(args.input)
    progress(f"Dimensions : {dict(ds.dims)}")

    sv        = ds["Sv"].values                    # (n_freq, n_ping, n_depth)
    depth     = ds["depth"].values                 # (n_depth,)
    freqs     = ds["channel"].values               # (n_freq,)
    time_dt   = ds["time"].values                  # datetime64[ns]
    lat       = ds["latitude"].values              # (n_ping,)
    lon       = ds["longitude"].values             # (n_ping,)
    day       = ds["day"].values.astype(np.int8)   # (n_ping,)
    ds.close()

    # Temps en secondes Unix (float64)
    time_unix = time_dt.astype("datetime64[s]").astype(np.float64)

    progress(f"  Sv shape : {sv.shape}  (freq, ping, depth)")

    # ------------------------------------------------------------------
    # 2. Binning temporel (200 s)
    # ------------------------------------------------------------------
    progress(f"Binning temporel : bins de {args.time_bin} s "
             f"(NaN si < {args.min_pings} pings) …")

    sv_t, time_bin, lat_bin, lon_bin, day_bin, counts = bin_sv_time(
        sv, time_unix, lat, lon, day,
        bin_sec=args.time_bin,
        min_pings=args.min_pings,
    )

    n_bins_valid = int((counts >= args.min_pings).sum())
    progress(f"  {sv_t.shape[1]} bins créés, {n_bins_valid} valides "
             f"({100 * n_bins_valid / sv_t.shape[1]:.1f} %)")

    # Reconvertir time en datetime64[ns]
    time_bin_dt = (time_bin * 1e9).astype("datetime64[ns]")

    # ------------------------------------------------------------------
    # 3. Binning spatial (10 m)
    # ------------------------------------------------------------------
    progress(f"Binning profondeur : bins de {args.depth_bin} m …")

    sv_out, depth_bin = bin_sv_depth(sv_t, depth, bin_m=args.depth_bin)

    progress(f"  {sv_out.shape[2]} bins de profondeur "
             f"({depth_bin[0]:.1f}–{depth_bin[-1]:.1f} m)")
    progress(f"  Sv shape final : {sv_out.shape}  (freq, time_bin, depth_bin)")

    # ------------------------------------------------------------------
    # 4. Construction du Dataset de sortie
    # ------------------------------------------------------------------
    progress("Construction du Dataset …")

    ds_out = xr.Dataset(
        {
            "Sv": xr.DataArray(
                sv_out,
                dims=["channel", "time", "depth"],
                attrs={"units": "dB re 1 m-1",
                       "long_name": "Volume backscattering strength (binned)"},
            ),
            "ping_count": xr.DataArray(
                counts,
                dims=["time"],
                attrs={"long_name": f"Number of pings in {args.time_bin}-s bin",
                       "min_pings_threshold": args.min_pings},
            ),
            "day": xr.DataArray(
                day_bin,
                dims=["time"],
                attrs={"long_name": "Day/Night flag (1=Night,2=Sunrise,3=Day,4=Sunset)"},
            ),
        },
        coords={
            "channel":   ("channel", freqs,
                          {"units": "kHz", "long_name": "Acoustic frequency"}),
            "time":      ("time", time_bin_dt),
            "depth":     ("depth", depth_bin,
                          {"units": "m", "long_name": "Depth (bin centre)",
                           "positive": "down"}),
            "latitude":  ("time", lat_bin,  {"units": "degrees_north"}),
            "longitude": ("time", lon_bin,  {"units": "degrees_east"}),
        },
    )

    ds_out.attrs.update({
        "time_bin_seconds":   args.time_bin,
        "depth_bin_meters":   args.depth_bin,
        "min_pings_per_bin":  args.min_pings,
        "sv_averaging":       "linear (10^(Sv/10)) then back to dB",
        "history":            f"Binned from {args.input}",
    })

    # ------------------------------------------------------------------
    # 5. Écriture
    # ------------------------------------------------------------------
    n_time_out  = ds_out.dims["time"]
    n_depth_out = ds_out.dims["depth"]

    encoding = {
        "Sv": {
            "zlib": True, "complevel": 4, "dtype": "float32",
            "chunksizes": (len(freqs), min(512, n_time_out), n_depth_out),
        },
        "time": {
            "units": "seconds since 1970-01-01",
            "calendar": "proleptic_gregorian",
            "dtype": "float64",
        },
        "ping_count": {"zlib": True, "dtype": "int32"},
        "day":        {"zlib": True, "dtype": "int8"},
        "latitude":   {"zlib": True, "complevel": 4, "dtype": "float64"},
        "longitude":  {"zlib": True, "complevel": 4, "dtype": "float64"},
    }

    progress(f"Écriture de {args.output} …")
    ds_out.to_netcdf(args.output, format="NETCDF4", encoding=encoding)

    progress(f"✓ Terminé : {args.output}")
    progress(
        f"  Dimensions finales : channel={len(freqs)}, "
        f"time={n_time_out}, depth={n_depth_out}"
    )


if __name__ == "__main__":
    main()