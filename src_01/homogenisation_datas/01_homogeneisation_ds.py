#!/usr/bin/env python3
"""
homogeineisation_ds.py — Filtre un fichier NetCDF acoustique (Sv) volumineux.

Filtres appliqués :
  - depth    : 15 m ≤ depth ≤ 1000 m
  - latitude : latitude < -40  (retrait des lats > 40°S)
  - longitude: longitude ≥ 50  (retrait des longitudes < 50°E)
  - variables: Sv, latitude, longitude, depth, time, channel,
               instrument_frequency, day

Masquage par fréquence (NaN au-delà) :
  - 18 kHz  : NaN pour depth > 995 m
  - 38 kHz  : NaN pour depth > 995 m
  - 70 kHz  : NaN pour depth > 595 m
  - 120 kHz : NaN pour depth > 395 m
  - 200 kHz : NaN pour depth > 195 m

Usage :
    python filter_acoustic.py \\
        --input  /chemin/vers/input.nc \\
        --output /chemin/vers/output.nc \\
        [--chunk-size 5000]
"""

import argparse
import sys
import time as _time

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Seuils de profondeur max valide par fréquence (kHz)
# ---------------------------------------------------------------------------
DEPTH_LIMITS_KHZ = {
    18:  995,
    38:  995,
    70:  595,
    120: 395,
    200: 195,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def progress(msg: str):
    print(f"[{_time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Filtre un fichier NetCDF acoustique (Sv) sans sA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      required=True,  help="Fichier NetCDF source (.nc)")
    parser.add_argument("--output",     required=True,  help="Fichier NetCDF filtré à créer (.nc)")
    parser.add_argument("--chunk-size", type=int, default=5000,
                        help="Taille des chunks dask sur la dimension time")
    parser.add_argument("--depth-min",  type=float, default=15.0,
                        help="Profondeur minimale (m)")
    parser.add_argument("--depth-max",  type=float, default=1000.0,
                        help="Profondeur maximale (m) — crop de l'axe depth")
    parser.add_argument("--lat-max",    type=float, default=-40.0,
                        help="Latitude maximale conservée (< lat-max, i.e. > 40°S)")
    parser.add_argument("--lon-min",    type=float, default=50.0,
                        help="Longitude minimale (°E)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Ouverture lazy avec dask
    # ------------------------------------------------------------------
    progress(f"Ouverture lazy de {args.input}")
    ds = xr.open_dataset(
        args.input,
        chunks={"time": args.chunk_size},
        mask_and_scale=True,
    )
    progress(f"Dimensions originales : {dict(ds.dims)}")

    # ------------------------------------------------------------------
    # 2. Crop depth (15 m → 1000 m)
    # ------------------------------------------------------------------
    depth_mask = (ds["depth"] >= args.depth_min) & (ds["depth"] <= args.depth_max)
    ds_filt = ds.sel(depth=depth_mask)

    n_depth_out = ds_filt.dims["depth"]
    progress(
        f"depth  : {ds.dims['depth']} → {n_depth_out} niveaux "
        f"({float(ds_filt['depth'].min()):.1f}–{float(ds_filt['depth'].max()):.1f} m)"
    )

    # ------------------------------------------------------------------
    # 3. Filtre spatial lat/lon
    # ------------------------------------------------------------------
    progress("Calcul du masque spatial (lat/lon) …")
    lat = ds_filt["latitude"].compute()
    lon = ds_filt["longitude"].compute()

    time_mask = (lat < args.lat_max) & (lon >= args.lon_min)
    ds_filt = ds_filt.sel(time=time_mask.values)

    n_time_in  = ds.dims["time"]
    n_time_out = ds_filt.dims["time"]
    progress(
        f"time   : {n_time_in} → {n_time_out} pings conservés "
        f"({100 * n_time_out / n_time_in:.1f} %)"
    )

    if n_time_out == 0:
        progress("⚠  Aucun ping ne satisfait les critères spatiaux. Abandon.")
        ds.close()
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Sélection des variables (sans sA)
    # ------------------------------------------------------------------
    vars_to_keep = ["Sv", "latitude", "longitude", "depth",
                    "time", "channel", "instrument_frequency", "day"]
    vars_present = [v for v in vars_to_keep if v in ds_filt]
    missing = set(vars_to_keep) - set(vars_present)
    if missing:
        progress(f"⚠  Variables absentes (ignorées) : {missing}")

    ds_out = ds_filt[vars_present].copy()

    # ------------------------------------------------------------------
    # 5. Masquage de Sv par fréquence au-delà des seuils de profondeur
    # ------------------------------------------------------------------
    if "Sv" in ds_out:
        progress("Application des masques NaN par fréquence …")

        freqs_khz = ds_out["channel"].values          # (n_freq,) en kHz
        depth_vals = ds_out["depth"].values            # (n_depth,)

        # Calcul du masque (channel, depth) → True = à mettre NaN
        # dims de Sv : (channel, time, depth)
        sv = ds_out["Sv"].values.copy()               # charge en RAM

        for f_idx, freq in enumerate(freqs_khz):
            # Trouver le seuil le plus proche dans le dictionnaire
            freq_key = min(DEPTH_LIMITS_KHZ.keys(), key=lambda k: abs(k - freq))
            depth_lim = DEPTH_LIMITS_KHZ[freq_key]

            nan_mask = depth_vals > depth_lim         # (n_depth,) booléen
            n_masked = nan_mask.sum()

            if n_masked > 0:
                # Broadcast sur (time,) : sv[f_idx, :, nan_mask] = NaN
                sv[f_idx, :, nan_mask] = np.nan
                progress(
                    f"  {int(freq):>6} kHz (→ seuil {depth_lim} m) : "
                    f"{n_masked} niveaux masqués"
                )

        # Remettre dans le dataset avec les mêmes attrs/dims
        ds_out["Sv"] = xr.DataArray(
            sv,
            dims=ds_out["Sv"].dims,
            coords=ds_out["Sv"].coords,
            attrs=ds_out["Sv"].attrs,
        )

    # ------------------------------------------------------------------
    # 6. Attributs globaux
    # ------------------------------------------------------------------
    ds_out.attrs.update({
        "filter_depth_min":      args.depth_min,
        "filter_depth_max":      args.depth_max,
        "filter_lat_max":        args.lat_max,
        "filter_lon_min":        args.lon_min,
        "freq_depth_limit_18kHz":  995,
        "freq_depth_limit_38kHz":  995,
        "freq_depth_limit_70kHz":  595,
        "freq_depth_limit_120kHz": 395,
        "freq_depth_limit_200kHz": 195,
    })

    # ------------------------------------------------------------------
    # 7. Encodage de sortie
    # ------------------------------------------------------------------
    encoding = {}
    for var in ds_out.data_vars:
        enc = {"zlib": True, "complevel": 4}
        if var in ds_filt:
            enc["dtype"] = ds_filt[var].dtype
        if var == "Sv" and ds_out[var].ndim == 3:
            dims = ds_out[var].dims
            csize = []
            for d in dims:
                if d == "time":
                    csize.append(min(512, n_time_out))
                elif d == "depth":
                    csize.append(n_depth_out)
                else:
                    csize.append(ds_out.dims[d])
            enc["chunksizes"] = tuple(csize)
        encoding[var] = enc

    for coord in ds_out.coords:
        if coord not in encoding:
            encoding[coord] = {"zlib": True, "complevel": 4}

    # ------------------------------------------------------------------
    # 8. Écriture
    # ------------------------------------------------------------------
    progress(f"Écriture de {args.output} …")
    ds_out.to_netcdf(
        args.output,
        format="NETCDF4",
        encoding=encoding,
        compute=True,
    )
    ds.close()

    progress(f"✓ Fichier filtré écrit : {args.output}")
    progress(
        f"  Dimensions finales : "
        f"channel={ds_out.dims.get('channel', '?')}, "
        f"time={n_time_out}, "
        f"depth={n_depth_out}"
    )


if __name__ == "__main__":
    main()