"""
bspline_fit.py
==============
Projection B-spline pénalisée de Sv par fréquence.
Sauvegarde un NetCDF par fréquence avec les coefficients et la reconstruction.

Nommage de sortie : {nom_input}_{freq}kHz_bspline.nc

Gestion de la profondeur :
  - crop automatique des bins de fond hors portée (NaN > 90% des pings)
  - filtre des profils avec trop de NaN (> max_nan)
  - phi, R, knots reconstruits sur depth_valid propre à chaque fréquence

Usage :
    python bspline_fit.py --input data.nc --output out/ --K 50 --r 4 --lambd 10
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.interpolate import BSpline
import scipy.linalg as la
import xarray as xr


# =============================================================================
# B-SPLINE
# =============================================================================

def build_knots(depth: np.ndarray, K: int, r: int) -> np.ndarray:
    """Construit le vecteur de noeuds étendu pour K bases d'ordre r."""
    return np.concatenate([
        np.repeat(depth[0],  r),
        np.linspace(depth[0], depth[-1], K - r + 2)[1:-1],
        np.repeat(depth[-1], r),
    ])


def build_phi(depth: np.ndarray, knots: np.ndarray, K: int, r: int) -> np.ndarray:
    """Matrice de base B-spline phi (n_depths, K)."""
    phi = np.zeros((len(depth), K))
    for k in range(K):
        c = np.zeros(K); c[k] = 1.0
        phi[:, k] = BSpline(knots, c, r - 1)(depth)
    return phi


def build_R(knots: np.ndarray, K: int, r: int,
            depth: np.ndarray, n_quad: int = 10_000) -> np.ndarray:
    """Matrice de pénalisation R = ∫ D²φ(z) D²φ(z)' dz  (K, K)."""
    z_quad = np.linspace(depth[0], depth[-1], n_quad)
    dz     = z_quad[1] - z_quad[0]
    D2     = np.zeros((n_quad, K))
    for k in range(K):
        c = np.zeros(K); c[k] = 1.0
        D2[:, k] = BSpline(knots, c, r - 1).derivative(2)(z_quad)
    return D2.T @ D2 * dz


# =============================================================================
# GESTION DES NaN ET CROP DE PROFONDEUR
# =============================================================================

def crop_depth(sv: np.ndarray, depth: np.ndarray,
               nan_col_thresh: float = 0.90) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Supprime les colonnes de profondeur (bins) dont plus de nan_col_thresh
    des pings sont NaN — typiquement le bas de colonne hors portée du sondeur.

    Retourne (sv_cropped, depth_valid, depth_mask).
    depth_mask : booléen (n_depths,), True = bin conservé.
    """
    nan_frac   = np.isnan(sv).mean(axis=0)   # (n_depths,)
    depth_mask = nan_frac < nan_col_thresh
    return sv[:, depth_mask], depth[depth_mask], depth_mask


def clean_sv(sv: np.ndarray, max_nan: float = 0.20,
             max_gap: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Garde les profils avec moins de max_nan de NaN global
    ET sans gap contigu de plus de max_gap * n_depths,
    puis interpole les NaN résiduels.
    Retourne (sv_clean, ping_mask).
    """
    n_depths  = sv.shape[1]
    max_gap_n = int(np.floor(max_gap * n_depths))   # nb de bins max autorisé

    # ── masque NaN global ─────────────────────────────────────────────────────
    nan_frac_mask = np.isnan(sv).mean(axis=1) < max_nan   # (n_pings,)

    # ── masque gap contigu ────────────────────────────────────────────────────
    def max_consecutive_nans(row: np.ndarray) -> int:
        """Longueur du plus grand gap contigu de NaN dans un profil."""
        max_gap_len = 0
        current     = 0
        for v in row:
            if np.isnan(v):
                current += 1
                max_gap_len = max(max_gap_len, current)
            else:
                current = 0
        return max_gap_len

    gap_mask = np.array([
        max_consecutive_nans(sv[i]) <= max_gap_n
        for i in range(sv.shape[0])
    ])   # (n_pings,)

    ping_mask = nan_frac_mask & gap_mask

    n_total   = sv.shape[0]
    n_kept    = ping_mask.sum()
    n_nan     = (~nan_frac_mask).sum()
    n_gap     = (nan_frac_mask & ~gap_mask).sum()
    print(f"  Profils supprimés — NaN global : {n_nan}  |  gap > {max_gap*100:.0f}% : {n_gap}  "
          f"|  conservés : {n_kept}/{n_total}")

    sv = sv[ping_mask].copy()

    # ── interpolation NaN résiduels ───────────────────────────────────────────
    for i in range(sv.shape[0]):
        row  = sv[i]
        nans = np.isnan(row)
        if nans.any():
            x = np.where(~nans)[0]
            if len(x) > 1:
                sv[i, nans] = np.interp(np.where(nans)[0], x, row[x])
            else:
                sv[i, nans] = np.nanmean(row)

    return sv, ping_mask


# =============================================================================
# AJUSTEMENT
# =============================================================================

def fit_bspline(sv: np.ndarray, phi: np.ndarray, R: np.ndarray,
                lambd: float) -> np.ndarray:
    """
    Résout α̂ = (Φ'Φ + λR)^{-1} Φ'y pour tous les profils.
    Retourne coefs (n_pings, K).
    """
    A    = phi.T @ phi + lambd * R   # (K, K)
    rhs  = phi.T @ sv.T              # (K, n_pings)
    return la.solve(A, rhs, assume_a="pos").T   # (n_pings, K)


# =============================================================================
# SAUVEGARDE NETCDF
# =============================================================================

def save_netcdf(ds_in: xr.Dataset,
                input_stem: str,
                freq: float,
                sv_clean: np.ndarray,
                coefs: np.ndarray,
                ping_mask: np.ndarray,
                depth_valid: np.ndarray,
                phi: np.ndarray,
                knots: np.ndarray,
                K: int, r: int, lambd: float,
                out_dir: Path) -> None:
    """Sauvegarde un NetCDF par fréquence.

    Nom de sortie : {input_stem}_{freq}kHz_bspline.nc
    depth est croppé à la plage valide pour cette fréquence.
    """

    time_full  = ds_in["time"].values
    time_valid = time_full[ping_mask]
    day_valid  = ds_in["day"].values[ping_mask]
    lat_valid  = ds_in["latitude"].values[ping_mask]
    lon_valid  = ds_in["longitude"].values[ping_mask]
    n_pings    = sv_clean.shape[0]

    sv_rec = (phi @ coefs.T).T   # (n_pings, n_depths_valid)

    bspline_k_coord = np.arange(K, dtype=np.int32)
    knot_coord      = np.arange(len(knots), dtype=np.int32)

    ds_out = xr.Dataset(
        {
            "Sv": xr.DataArray(
                sv_clean,
                dims=["time", "depth"],
                coords={"time": time_valid, "depth": depth_valid},
                attrs={"long_name": "Sv original (profils valides)", "units": "dB re 1 m-1"},
            ),
            "Sv_reconstructed": xr.DataArray(
                sv_rec,
                dims=["time", "depth"],
                coords={"time": time_valid, "depth": depth_valid},
                attrs={"long_name": "Sv reconstruit B-spline", "units": "dB re 1 m-1"},
            ),
            "bspline_coeffs": xr.DataArray(
                coefs,
                dims=["time", "bspline_k"],
                coords={"time": time_valid, "bspline_k": bspline_k_coord},
                attrs={"long_name": "Coefficients B-spline"},
            ),
            "bspline_knots": xr.DataArray(
                knots,
                dims=["knot"],
                coords={"knot": knot_coord},
                attrs={"long_name": "Noeuds B-spline etendus"},
            ),
            "day": xr.DataArray(
                day_valid,
                dims=["time"],
                coords={"time": time_valid},
                attrs=ds_in["day"].attrs,
            ),
            "latitude": xr.DataArray(
                lat_valid,
                dims=["time"],
                coords={"time": time_valid},
                attrs=ds_in["latitude"].attrs,
            ),
            "longitude": xr.DataArray(
                lon_valid,
                dims=["time"],
                coords={"time": time_valid},
                attrs=ds_in["longitude"].attrs,
            ),
        },
        attrs={
            "frequency_kHz":  float(freq) / 1000 if freq > 1000 else float(freq),
            "bspline_K":      int(K),
            "bspline_r":      int(r),
            "bspline_d":      int(r),
            "bspline_lambda": float(lambd),
            "n_pings_total":  int(len(time_full)),
            "n_pings_valid":  int(n_pings),
            "depth_min":      float(depth_valid[0]),
            "depth_max":      float(depth_valid[-1]),
            "n_depths_valid": int(len(depth_valid)),
        },
    )

    # Copier les attributs globaux du dataset source
    ds_out.attrs.update({k: v for k, v in ds_in.attrs.items()
                         if k not in ds_out.attrs})

    # Nom de sortie : {input_stem}_{freq}kHz_bspline.nc
    freq_khz = freq / 1000 if freq > 1000 else freq
    freq_str = f"{freq_khz:.0f}"
    out_path = out_dir / f"{input_stem}_{freq_str}kHz_bspline.nc"

    ds_out.to_netcdf(out_path, format="NETCDF4",
                     encoding={
                         "Sv":               {"zlib": True, "complevel": 4},
                         "Sv_reconstructed": {"zlib": True, "complevel": 4},
                         "bspline_coeffs":   {"zlib": True, "complevel": 4},
                     })
    print(f"  Sauvegardé : {out_path}  (depth [{depth_valid[0]:.0f}–{depth_valid[-1]:.0f}] m)")


# =============================================================================
# PIPELINE PAR FRÉQUENCE
# =============================================================================

def run(path_in: str, path_out: str,
        K: int = 50, r: int = 4, lambd: float = 10.0,
        max_nan: float = 0.20,
        nan_col_thresh: float = 0.90) -> None:

    out_dir    = Path(path_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_stem = Path(path_in).stem

    print("=" * 55)
    print("B-SPLINE FIT  —  K={K}  r={r}  λ={lambd}".format(**locals()))
    print("=" * 55)

    with xr.open_dataset(path_in) as ds:
        depth_full   = ds["depth"].values
        channel_vals = ds["channel"].values

        print(f"  depth full : [{depth_full[0]:.0f}, {depth_full[-1]:.0f}] m  "
              f"({len(depth_full)} points)")

        for freq_idx, freq in enumerate(channel_vals):
            freq = float(freq)
            print(f"\n[{freq_idx+1}/{len(channel_vals)}] {freq:.0f} kHz")

            sv_full = ds["Sv"].values[freq_idx]   # (n_pings, n_depths)

            # ── 1. Crop profondeur : retire les bins hors portée ─────────
            sv_cropped, depth_valid, _ = crop_depth(
                sv_full, depth_full, nan_col_thresh=nan_col_thresh
            )
            print(f"  depth valide : [{depth_valid[0]:.0f}, {depth_valid[-1]:.0f}] m  "
                  f"({len(depth_valid)} bins / {len(depth_full)})")

            # ── 2. Filtre et interpolation des NaN résiduels par profil ──
            sv_clean, ping_mask = clean_sv(sv_cropped, max_nan=max_nan)
            print(f"  Profils valides : {sv_clean.shape[0]}/{sv_full.shape[0]} "
                  f"(nan < {max_nan*100:.0f}%)")

            # ── 3. Matrices B-spline sur depth_valid ─────────────────────
            knots = build_knots(depth_valid, K, r)
            phi   = build_phi(depth_valid, knots, K, r)
            R     = build_R(knots, K, r, depth_valid)
            print(f"  K={K}  len(knots)={len(knots)}  attendu={K+r}")

            # ── 4. Fit ────────────────────────────────────────────────────
            coefs = fit_bspline(sv_clean, phi, R, lambd)
            print(f"  coefs shape : {coefs.shape}")

            # ── 5. Sauvegarde ─────────────────────────────────────────────
            save_netcdf(ds, input_stem, freq, sv_clean, coefs,
                        ping_mask, depth_valid, phi, knots, K, r, lambd, out_dir)

    print("\n✓ Terminé.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Projection B-spline pénalisée de Sv — un NetCDF par fréquence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",          required=True,          help="NetCDF source")
    parser.add_argument("--output",         default=".",            help="Répertoire de sortie")
    parser.add_argument("--K",              type=int,   default=50,   help="Nb de bases B-spline")
    parser.add_argument("--r",              type=int,   default=4,    help="Ordre B-spline")
    parser.add_argument("--lambd",          type=float, default=10.0, help="Paramètre de pénalisation λ")
    parser.add_argument("--max_nan",        type=float, default=0.20, help="Seuil NaN par profil (0–1)")
    parser.add_argument("--nan_col_thresh", type=float, default=0.90, help="Seuil NaN par bin de profondeur (0–1)")
    args = parser.parse_args()

    run(
        path_in        = args.input,
        path_out       = args.output,
        K              = args.K,
        r              = args.r,
        lambd          = args.lambd,
        max_nan        = args.max_nan,
        nan_col_thresh = args.nan_col_thresh,
    )