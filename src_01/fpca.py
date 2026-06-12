"""
fpca_bspline.py
===============
FPCA sur les coefficients B-spline d'un fichier NetCDF par fréquence
produit par bspline_fit.py.

Entrée  : {stem}_{freq}kHz_bspline.nc
Sortie  : {stem}_{freq}kHz_bspline_fpca.nc

Usage :
    python fpca_bspline.py --input Echointegration2022_38kHz_bspline.nc --output out/
    python fpca_bspline.py --input out/*.nc --output out/   # glob
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.interpolate import BSpline
import scipy.linalg as la
import xarray as xr


# =============================================================================
# RECONSTRUCTION DE PHI (base B-spline sur la grille de profondeur)
# =============================================================================

def build_phi(depth: np.ndarray, knots: np.ndarray, K: int, r: int) -> np.ndarray:
    """Matrice de base B-spline phi (n_depths, K)."""
    phi = np.zeros((len(depth), K))
    for k in range(K):
        c = np.zeros(K); c[k] = 1.0
        phi[:, k] = BSpline(knots, c, r - 1)(depth)
    return phi


# =============================================================================
# MATRICE DE GRAM  W[j,k] = ∫ B_j(z) · B_k(z) dz
# =============================================================================

def build_gram(knots: np.ndarray, K: int, r: int,
               depth: np.ndarray, n_quad: int = 10_000) -> np.ndarray:
    """
    Matrice de Gram W (K, K) par quadrature numérique.
    Équivalent de eval.penalty en R (sans pénalisation, juste le produit scalaire L²).
    """
    z_quad = np.linspace(depth[0], depth[-1], n_quad)
    dz     = z_quad[1] - z_quad[0]

    phi_quad = np.zeros((n_quad, K))
    for k in range(K):
        c = np.zeros(K); c[k] = 1.0
        phi_quad[:, k] = BSpline(knots, c, r - 1)(z_quad)

    W = phi_quad.T @ phi_quad * dz   # (K, K)
    return (W + W.T) / 2             # symétrie explicite


# =============================================================================
# FPCA
# =============================================================================

def run_fpca(coefs: np.ndarray, W: np.ndarray) -> dict:
    """
    FPCA sur les coefficients B-spline.

    Paramètres
    ----------
    coefs : (n_pings, K)  — coefficients B-spline
    W     : (K, K)        — matrice de Gram

    Retourne un dict avec :
        coeffs_mean  (K,)
        C            (n_pings, K)   — coefficients centrés
        eigenvalues  (K,)           — valeurs propres triées décroissant
        vectors      (K, K)         — vecteurs propres W-normalisés
        axe          (K, K)         — vecteurs * sqrt(eigenvalues), déformation ±1σ
        scores       (n_pings, K)   — scores FPCA  C @ W @ vectors
        var_ratio    (K,)           — variance expliquée par composante (fraction)
        cum_var      (K,)           — variance cumulée
    """
    n_pings, K = coefs.shape

    # ── Centrage ─────────────────────────────────────────────────────────────
    coeffs_mean = np.nanmean(coefs, axis=0)          # (K,)
    C           = coefs - coeffs_mean                # (n_pings, K)
    covar       = (C.T @ C) / n_pings               # (K, K)

    # ── Cholesky de W  →  W = Wdem.T @ Wdem  (upper, comme R) ───────────────
    Wdem    = la.cholesky(W, lower=False)                          # (K, K)
    Wdeminv = la.solve_triangular(Wdem, np.eye(K), lower=False)   # W^{-1/2}

    # ── Valeurs / vecteurs propres de  Wdem @ covar @ Wdem.T ─────────────────
    covarW             = Wdem @ covar @ Wdem.T
    eigenvalues, eigvec = la.eigh(covarW)            # eigh car symétrique

    # Tri décroissant
    order        = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[order]
    eigvec       = eigvec[:, order]

    # ── Retour dans l'espace original ────────────────────────────────────────
    vectors = Wdeminv @ eigvec                                         # (K, K)
    axe     = vectors * np.sqrt(np.maximum(eigenvalues, 0))[np.newaxis, :]  # (K, K)

    # ── Scores ───────────────────────────────────────────────────────────────
    scores = C @ W @ vectors                          # (n_pings, K)

    # ── Variance expliquée ───────────────────────────────────────────────────
    total     = eigenvalues.sum()
    var_ratio = eigenvalues / total if total > 0 else np.zeros(K)
    cum_var   = np.cumsum(var_ratio)

    return dict(
        coeffs_mean = coeffs_mean,
        C           = C,
        eigenvalues = eigenvalues,
        vectors     = vectors,
        axe         = axe,
        scores      = scores,
        var_ratio   = var_ratio,
        cum_var     = cum_var,
    )


# =============================================================================
# SAUVEGARDE
# =============================================================================

def save_fpca_netcdf(ds_in: xr.Dataset, fpca: dict,
                     phi: np.ndarray, out_path: Path) -> None:
    """
    Enrichit le dataset d'entrée avec les résultats FPCA et sauvegarde.

    Variables ajoutées
    ------------------
    fpca_mean_coeff     (bspline_k,)              — coefficients moyens
    fpca_mean_profile   (depth,)                  — profil moyen reconstruit
    fpca_eigenvalues    (fpca_component,)
    fpca_var_ratio      (fpca_component,)
    fpca_cumulative_var (fpca_component,)
    fpca_vectors        (bspline_k, fpca_component) — vecteurs W-normalisés
    fpca_axe            (bspline_k, fpca_component) — déformation ±1σ
    fpca_eigenfunctions (depth, fpca_component)   — φ(z) = phi @ vectors
    fpca_scores         (time, fpca_component)
    """
    K          = fpca["vectors"].shape[0]
    n_comp     = K
    comp_coord = np.arange(n_comp, dtype=np.int32)

    depth      = ds_in["depth"].values
    time       = ds_in["time"].values
    k_coord    = ds_in["bspline_k"].values

    # Eigenfunctions dans l'espace profondeur : (depth, component)
    eigenfunctions = phi @ fpca["vectors"]   # (n_depths, K)

    ds_out = ds_in.copy()   # conserve Sv, Sv_reconstructed, bspline_coeffs, bspline_knots

    ds_out = ds_out.assign({

        "fpca_mean_coeff": xr.DataArray(
            fpca["coeffs_mean"],
            dims=["bspline_k"],
            coords={"bspline_k": k_coord},
            attrs={"long_name": "Coefficients B-spline moyens"},
        ),

        "fpca_mean_profile": xr.DataArray(
            phi @ fpca["coeffs_mean"],
            dims=["depth"],
            coords={"depth": depth},
            attrs={"long_name": "Profil Sv moyen reconstruit", "units": "dB re 1 m-1"},
        ),

        "fpca_eigenvalues": xr.DataArray(
            fpca["eigenvalues"],
            dims=["fpca_component"],
            coords={"fpca_component": comp_coord},
            attrs={"long_name": "Valeurs propres FPCA"},
        ),

        "fpca_var_ratio": xr.DataArray(
            fpca["var_ratio"],
            dims=["fpca_component"],
            coords={"fpca_component": comp_coord},
            attrs={"long_name": "Variance expliquée par composante (fraction)"},
        ),

        "fpca_cumulative_var": xr.DataArray(
            fpca["cum_var"],
            dims=["fpca_component"],
            coords={"fpca_component": comp_coord},
            attrs={"long_name": "Variance cumulée expliquée (fraction)"},
        ),

        "fpca_vectors": xr.DataArray(
            fpca["vectors"],
            dims=["bspline_k", "fpca_component"],
            coords={"bspline_k": k_coord, "fpca_component": comp_coord},
            attrs={"long_name": "Vecteurs propres W-normalisés (espace coefficients)"},
        ),

        "fpca_axe": xr.DataArray(
            fpca["axe"],
            dims=["bspline_k", "fpca_component"],
            coords={"bspline_k": k_coord, "fpca_component": comp_coord},
            attrs={"long_name": "Déformation ±1σ (vectors * sqrt(eigenvalues))"},
        ),

        "fpca_eigenfunctions": xr.DataArray(
            eigenfunctions,
            dims=["depth", "fpca_component"],
            coords={"depth": depth, "fpca_component": comp_coord},
            attrs={"long_name": "Eigenfunctions FPCA dans l'espace profondeur"},
        ),

        "fpca_scores": xr.DataArray(
            fpca["scores"],
            dims=["time", "fpca_component"],
            coords={"time": time, "fpca_component": comp_coord},
            attrs={"long_name": "Scores FPCA par ping"},
        ),
    })

    # Attributs globaux
    n_comp_90 = int(np.searchsorted(fpca["cum_var"], 0.90)) + 1
    ds_out.attrs["fpca_n_components_total"] = int(K)
    ds_out.attrs["fpca_n_components_90pct"] = int(n_comp_90)

    ds_out.to_netcdf(
        out_path, format="NETCDF4",
        encoding={
            "Sv":                  {"zlib": True, "complevel": 4},
            "Sv_reconstructed":    {"zlib": True, "complevel": 4},
            "bspline_coeffs":      {"zlib": True, "complevel": 4},
            "fpca_scores":         {"zlib": True, "complevel": 4},
            "fpca_eigenfunctions": {"zlib": True, "complevel": 4},
        },
    )
    print(f"  Sauvegardé : {out_path}")


# =============================================================================
# PIPELINE
# =============================================================================

def run(path_in: str, path_out: str, n_quad: int = 10_000) -> None:

    out_dir = Path(path_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(path_in)
    print("=" * 55)
    print(f"FPCA B-SPLINE  —  {input_path.name}")
    print("=" * 55)

    with xr.open_dataset(path_in) as ds:

        # ── Lecture des métadonnées B-spline ──────────────────────────────
        K     = int(ds.attrs["bspline_K"])
        r     = int(ds.attrs["bspline_r"])
        depth = ds["depth"].values
        knots = ds["bspline_knots"].values
        coefs = ds["bspline_coeffs"].values   # (n_pings, K)

        n_pings, _ = coefs.shape
        freq_kHz   = ds.attrs.get("frequency_kHz", "?")
        print(f"  Fréquence   : {freq_kHz} kHz")
        print(f"  Pings       : {n_pings}")
        print(f"  K={K}  r={r}  depth=[{depth[0]:.0f}, {depth[-1]:.0f}] m")

        # ── Matrices B-spline ─────────────────────────────────────────────
        phi = build_phi(depth, knots, K, r)
        W   = build_gram(knots, K, r, depth, n_quad=n_quad)
        print(f"  phi : {phi.shape}   W : {W.shape}")

        # ── FPCA ──────────────────────────────────────────────────────────
        fpca = run_fpca(coefs, W)

        n90 = int(np.searchsorted(fpca["cum_var"], 0.90)) + 1
        print(f"  Composantes pour 90% var : {n90}")
        for i in range(min(n90, 10)):
            print(f"    PC{i+1}  {fpca['var_ratio'][i]*100:.2f}%  "
                  f"(cum. {fpca['cum_var'][i]*100:.1f}%)")

        # ── Sauvegarde ────────────────────────────────────────────────────
        # Nom : {stem_sans_bspline}_{freq}kHz_bspline_fpca.nc
        out_path = out_dir / (input_path.stem + "_fpca.nc")
        save_fpca_netcdf(ds, fpca, phi, out_path)

    print("\n✓ Terminé.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FPCA sur les coefficients B-spline d'un NetCDF par fréquence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  required=True, help="NetCDF source (*_bspline.nc)")
    parser.add_argument("--output", default=".",   help="Répertoire de sortie")
    parser.add_argument("--n_quad", type=int, default=10_000,
                        help="Points de quadrature pour la matrice de Gram")
    args = parser.parse_args()

    run(path_in=args.input, path_out=args.output, n_quad=args.n_quad)