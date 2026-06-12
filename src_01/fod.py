import numpy as np
import xarray as xr
from scipy.linalg import eigh
from scipy.interpolate import BSpline
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def compute_bspline_gram_matrix(knots, K, order, n_quad=500):
    z_min, z_max = knots[0], knots[-1]
    z_quad = np.linspace(z_min, z_max, n_quad)
    Phi = np.zeros((n_quad, K))
    for k in range(K):
        c = np.zeros(K)
        c[k] = 1.0
        spl = BSpline(knots, c, order - 1)
        Phi[:, k] = spl(z_quad)
    dz = (z_max - z_min) / (n_quad - 1)
    return Phi.T @ Phi * dz


def _matrix_sqrt(A, inv=False):
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 1e-12)
    s = 1.0 / np.sqrt(vals) if inv else np.sqrt(vals)
    return vecs @ np.diag(s) @ vecs.T


def eval_bspline_basis(knots, K, order, z_eval):
    """Évalue toutes les fonctions de base B-spline sur z_eval → (n_z, K)."""
    Phi = np.zeros((len(z_eval), K))
    for k in range(K):
        c = np.zeros(K)
        c[k] = 1.0
        spl = BSpline(knots, c, order - 1)
        Phi[:, k] = spl(z_eval)
    return Phi


# ─────────────────────────────────────────────────────────────────────────────
# FPCA BIVARIÉE
# ─────────────────────────────────────────────────────────────────────────────

def fpca_bivariate_bspline(path_in):
    ds = xr.open_dataset(path_in)

    C_T     = ds["thetao_bspline_coeffs"].values   # (N_time, N_lat, N_lon, K)
    C_S     = ds["so_bspline_coeffs"].values
    knots_T = ds["bspline_knots"].values
    knots_S = ds["bspline_knots"].values
    order   = int(ds.attrs.get("bspline_thetao_d", 4))
    lats    = ds["latitude"].values
    lons    = ds["longitude"].values
    times   = ds["time"].values
    depths  = ds["depth"].values
    ds.close()

    N_time, N_lat, N_lon, K = C_T.shape

    # reshape (N_pixels * N_time, K)
    C_T_flat = C_T.transpose(1, 2, 0, 3).reshape(-1, K)
    C_S_flat = C_S.transpose(1, 2, 0, 3).reshape(-1, K)

    valid = np.isfinite(C_T_flat).all(axis=1) & np.isfinite(C_S_flat).all(axis=1)
    C_T_v = C_T_flat[valid]
    C_S_v = C_S_flat[valid]
    N = C_T_v.shape[0]

    X          = np.concatenate([C_T_v, C_S_v], axis=1)   # (N, 2K)
    alpha_mean = X.mean(axis=0)
    Xc         = X - alpha_mean                             # centré

    V    = (Xc.T @ Xc) / (N - 1)
    V_TT = V[:K, :K]
    V_SS = V[K:, K:]

    W_T = compute_bspline_gram_matrix(knots_T, K, order)
    W_S = compute_bspline_gram_matrix(knots_S, K, order)
    W   = np.block([[W_T, np.zeros((K, K))],
                    [np.zeros((K, K)), W_S]])

    sigma2_T = np.trace(V_TT @ W_T)
    sigma2_S = np.trace(V_SS @ W_S)
    m        = np.concatenate([np.full(K, 1.0 / sigma2_T),
                                np.full(K, 1.0 / sigma2_S)])

    M_sqrt     = np.diag(np.sqrt(m))
    W_sqrt     = _matrix_sqrt(W)
    A          = M_sqrt @ W_sqrt @ V @ W_sqrt @ M_sqrt

    eigenvalues, B = eigh(A)
    eigenvalues    = eigenvalues[::-1]
    B              = B[:, ::-1]

    M_sqrt_inv = np.diag(1.0 / np.sqrt(m))
    W_sqrt_inv = _matrix_sqrt(W, inv=True)
    Beta       = M_sqrt_inv @ W_sqrt_inv @ B    # (2K, 2K)

    WM     = W @ np.diag(m)
    scores = Xc @ WM @ Beta                     # (N, 2K)

    n_comp     = min(2 * K, 18)
    eigenfuncs = np.stack([Beta[:K, :n_comp].T,
                           Beta[K:, :n_comp].T], axis=1)   # (n_comp, 2, K)

    return {
        "scores":      scores[:, :n_comp],
        "eigenfuncs":  eigenfuncs,
        "eigenvalues": eigenvalues[:n_comp],
        "alpha_mean":  alpha_mean,
        "valid_mask":  valid,
        "sigma2_T":    sigma2_T,
        "sigma2_S":    sigma2_S,
        "knots_T":     knots_T,
        "knots_S":     knots_S,
        "order":       order,
        "depths":      depths,
        "lats":        lats,
        "lons":        lons,
        "times":       times,
        "N_time":      N_time,
        "N_lat":       N_lat,
        "N_lon":       N_lon,
        "K":           K,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GMM CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def gmm_clustering_fpca(result, var_threshold=0.90, n_clusters=None):
    eigenvalues = result["eigenvalues"]
    scores      = result["scores"]
    valid_mask  = result["valid_mask"]
    N_time      = result["N_time"]
    N_lat       = result["N_lat"]
    N_lon       = result["N_lon"]

    # sélection des composantes à var_threshold
    var_ratio = eigenvalues / eigenvalues.sum()
    cum_var   = np.cumsum(var_ratio)
    K_sel     = int(np.searchsorted(cum_var, var_threshold)) + 1
    print(f"Composantes retenues : {K_sel}  ({100*cum_var[K_sel-1]:.1f}% variance)")

    scores_sel = scores[:, :K_sel]
    scaler     = StandardScaler()
    scores_std = scaler.fit_transform(scores_sel)

    # BIC si n_clusters non fourni
    bics = None
    def _elbow_index(values):
        """
        Retourne l'indice du coude dans une courbe décroissante
        par distance perpendiculaire maximale à la droite [premier, dernier point].
        """
        n   = len(values)
        pts = np.array([[i, values[i]] for i in range(n)], dtype=float)

        # normalisation pour que x et y soient comparables
        pts[:, 0] /= (n - 1)
        pts[:, 1]  = (pts[:, 1] - pts[:, 1].min()) / (pts[:, 1].max() - pts[:, 1].min() + 1e-12)

        # droite entre premier et dernier point
        p1, p2 = pts[0], pts[-1]
        d      = p2 - p1
        d_norm = d / np.linalg.norm(d)

        # distance perpendiculaire de chaque point à la droite
        perp_dists = np.abs(np.cross(d_norm, pts - p1))

        return int(np.argmax(perp_dists))


    if n_clusters is None:
        k_range = range(2, 11)
        bics    = []
        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type="full",
                                random_state=42, n_init=5)
            gmm.fit(scores_std)
            bics.append(gmm.bic(scores_std))

        elbow_idx  = _elbow_index(bics)
        n_clusters = list(k_range)[elbow_idx]
        print(f"Clusters optimal (coude BIC) : {n_clusters}  "
            f"(minimum absolu : {list(k_range)[int(np.argmin(bics))]})")

    gmm    = GaussianMixture(n_components=n_clusters, covariance_type="full",
                             random_state=42, n_init=10)
    labels = gmm.fit_predict(scores_std)    # (N_valid,)
    proba  = gmm.predict_proba(scores_std)  # (N_valid, n_clusters)

    # remise sur (N_pixels * N_time)
    N_pixels   = N_lat * N_lon
    labels_full = np.full(valid_mask.shape[0], -1, dtype=int)
    proba_full  = np.full((valid_mask.shape[0], n_clusters), np.nan)
    labels_full[valid_mask] = labels
    proba_full[valid_mask]  = proba

    # reshape (N_pixels, N_time) → mode par pixel
    labels_pt = labels_full.reshape(N_pixels, N_time)
    proba_pt  = proba_full.reshape(N_pixels, N_time, n_clusters)

    labels_pixel = np.full(N_pixels, -1, dtype=int)
    proba_pixel  = np.full((N_pixels, n_clusters), np.nan)
    for i in range(N_pixels):
        row = labels_pt[i][labels_pt[i] >= 0]
        if len(row) > 0:
            labels_pixel[i]  = stats.mode(row, keepdims=True).mode[0]
            proba_pixel[i]   = np.nanmean(proba_pt[i][labels_pt[i] >= 0], axis=0)

    labels_map = labels_pixel.reshape(N_lat, N_lon).astype(float)
    proba_map  = proba_pixel.reshape(N_lat, N_lon, n_clusters)
    labels_map[labels_map < 0] = np.nan

    # scores sur grille (N_pixels, N_time, K_sel) → (N_lat, N_lon, N_time, K_sel)
    scores_full = np.full((valid_mask.shape[0], K_sel), np.nan)
    scores_full[valid_mask] = scores_sel
    scores_map = scores_full.reshape(N_pixels, N_time, K_sel
                                     ).reshape(N_lat, N_lon, N_time, K_sel)

    return {
        "labels_map":  labels_map,       # (lat, lon)
        "proba_map":   proba_map,        # (lat, lon, n_clusters)
        "scores_map":  scores_map,       # (lat, lon, time, K_sel)
        "n_clusters":  n_clusters,
        "K_sel":       K_sel,
        "cum_var":     cum_var,
        "var_ratio":   var_ratio,
        "bics":        bics,
        "scaler":      scaler,
        "gmm":         gmm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE NETCDF
# ─────────────────────────────────────────────────────────────────────────────

def save_results_netcdf(path_in, result_fpca, result_gmm, path_out=None):
    if path_out is None:
        base     = os.path.splitext(path_in)[0]
        path_out = base + "_fpca_gmm.nc"

    lats       = result_fpca["lats"]
    lons       = result_fpca["lons"]
    times      = result_fpca["times"]
    depths     = result_fpca["depths"]
    knots_T    = result_fpca["knots_T"]
    knots_S    = result_fpca["knots_S"]
    order      = result_fpca["order"]
    K          = result_fpca["K"]
    n_comp     = result_fpca["eigenfuncs"].shape[0]   # 18 — toutes les composantes
    n_clusters = result_gmm["n_clusters"]
    K_sel      = result_gmm["K_sel"]                  #  2 — composantes retenues

    # eigenfunctions évaluées sur la grille de profondeur
    Phi_T = eval_bspline_basis(knots_T, K, order, depths)
    Phi_S = eval_bspline_basis(knots_S, K, order, depths)

    eigfunc_T = result_fpca["eigenfuncs"][:, 0, :] @ Phi_T.T  # (n_comp, n_depth)
    eigfunc_S = result_fpca["eigenfuncs"][:, 1, :] @ Phi_S.T

    alpha_mean = result_fpca["alpha_mean"]
    mu_T = alpha_mean[:K] @ Phi_T.T
    mu_S = alpha_mean[K:] @ Phi_S.T

    ds_out = xr.Dataset(
        {
            # ── FPCA scores (K_sel composantes retenues) ──────────────────
            # dim "fpca_component_kept" ≠ "fpca_component" → pas de conflit
            "fpca_scores": xr.DataArray(
                result_gmm["scores_map"],
                dims=["latitude", "longitude", "time", "fpca_component_kept"],
                attrs={"long_name": "FPCA bivariate scores", "units": "1"},
            ),

            # ── FPCA spectre (n_comp composantes totales) ─────────────────
            "fpca_eigenvalues": xr.DataArray(
                result_fpca["eigenvalues"],
                dims=["fpca_component"],
                attrs={"long_name": "FPCA eigenvalues"},
            ),
            "fpca_variance_ratio": xr.DataArray(
                result_gmm["var_ratio"][:n_comp],
                dims=["fpca_component"],
                attrs={"long_name": "Fraction of variance explained per component"},
            ),
            "fpca_cumvar": xr.DataArray(
                result_gmm["cum_var"][:n_comp],
                dims=["fpca_component"],
                attrs={"long_name": "Cumulative variance explained"},
            ),

            # ── Eigenfunctions (n_comp composantes totales) ───────────────
            "eigenfunction_thetao": xr.DataArray(
                eigfunc_T,
                dims=["fpca_component", "depth"],
                attrs={"long_name": "Eigenfunction — temperature component",
                       "units": "°C"},
            ),
            "eigenfunction_so": xr.DataArray(
                eigfunc_S,
                dims=["fpca_component", "depth"],
                attrs={"long_name": "Eigenfunction — salinity component",
                       "units": "psu"},
            ),

            # ── Profils moyens ────────────────────────────────────────────
            "mean_profile_thetao": xr.DataArray(
                mu_T,
                dims=["depth"],
                attrs={"long_name": "Mean temperature profile", "units": "°C"},
            ),
            "mean_profile_so": xr.DataArray(
                mu_S,
                dims=["depth"],
                attrs={"long_name": "Mean salinity profile", "units": "psu"},
            ),

            # ── GMM ───────────────────────────────────────────────────────
            "gmm_cluster": xr.DataArray(
                result_gmm["labels_map"],
                dims=["latitude", "longitude"],
                attrs={"long_name": "GMM cluster index (modal over time)",
                       "units": "1"},
            ),
            "gmm_proba": xr.DataArray(
                result_gmm["proba_map"],
                dims=["latitude", "longitude", "cluster"],
                attrs={"long_name": "GMM posterior probability per cluster",
                       "units": "1"},
            ),
        },
        coords={
            "latitude":              ("latitude",              lats),
            "longitude":             ("longitude",             lons),
            "time":                  ("time",                  times),
            "depth":                 ("depth",                 depths),
            "fpca_component":        ("fpca_component",        np.arange(n_comp)),
            "fpca_component_kept":   ("fpca_component_kept",   np.arange(K_sel)),
            "cluster":               ("cluster",               np.arange(n_clusters)),
        },
        attrs={
            "description":       "FPCA bivariée (T, S) + clustering GMM",
            "fpca_K_bspline":    int(K),
            "fpca_order":        int(order),
            "fpca_n_comp_total": int(n_comp),
            "fpca_K_selected":   int(K_sel),
            "gmm_n_clusters":    int(n_clusters),
            "gmm_covariance":    "full",
        },
    )

    ds_out.to_netcdf(path_out)
    print(f"Sauvegardé : {path_out}")
    return path_out


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_map(result_fpca, result_gmm, path_in):
    labels_map = result_gmm["labels_map"]
    n_clusters = result_gmm["n_clusters"]
    K_sel      = result_gmm["K_sel"]
    cum_var    = result_gmm["cum_var"]
    lats       = result_fpca["lats"]
    lons       = result_fpca["lons"]

    ds = xr.open_dataset(path_in)
    thetao_surf = ds["thetao"].isel(depth=0).mean(dim="time").values
    ds.close()

    cmap_disc = mcolors.ListedColormap(plt.cm.tab10(np.linspace(0, 1, n_clusters)))
    norm      = mcolors.BoundaryNorm(np.arange(-0.5, n_clusters + 0.5), cmap_disc.N)

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": proj})

    ax.pcolormesh(lons, lats, labels_map, cmap=cmap_disc, norm=norm,
                  transform=proj, alpha=0.85, shading="auto")

    cs = ax.contour(lons, lats, thetao_surf, levels=10,
                    colors="k", linewidths=0.6, alpha=0.5, transform=proj)
    ax.clabel(cs, fmt="%.1f°C", fontsize=7, inline=True)

    ax.add_feature(cfeature.LAND,      facecolor="lightgray", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8,          zorder=4)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    sm   = plt.cm.ScalarMappable(cmap=cmap_disc, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, ticks=np.arange(n_clusters),
                        pad=0.02, fraction=0.03)
    cbar.set_ticklabels([f"Cluster {k}" for k in range(n_clusters)])

    ax.set_title(
        f"GMM ({n_clusters} clusters) — FPCA bivariée T/S\n"
        f"{K_sel} composantes ({100*cum_var[K_sel-1]:.0f}% variance)",
        fontsize=12,
    )
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=proj)
    plt.tight_layout()
    plt.show()


def plot_bic(result_gmm):
    bics = result_gmm.get("bics")
    if bics is None:
        print("BIC non disponible (n_clusters fourni manuellement).")
        return
    k_range = range(2, 2 + len(bics))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(list(k_range), bics, "o-", color="steelblue")
    ax.axvline(result_gmm["n_clusters"], color="coral", ls="--",
               label=f"optimal k={result_gmm['n_clusters']}")
    ax.set_xlabel("Nombre de clusters")
    ax.set_ylabel("BIC")
    ax.set_title("Sélection du nombre de clusters")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_eigenfunctions(result_fpca, n_modes=3):
    depths  = result_fpca["depths"]
    knots_T = result_fpca["knots_T"]
    knots_S = result_fpca["knots_S"]
    order   = result_fpca["order"]
    K       = result_fpca["K"]
    alpha_mean  = result_fpca["alpha_mean"]
    eigenvalues = result_fpca["eigenvalues"]

    Phi_T = eval_bspline_basis(knots_T, K, order, depths)
    Phi_S = eval_bspline_basis(knots_S, K, order, depths)
    mu_T  = alpha_mean[:K] @ Phi_T.T
    mu_S  = alpha_mean[K:] @ Phi_S.T

    fig, axes = plt.subplots(n_modes, 2, figsize=(10, 4 * n_modes))
    for k in range(n_modes):
        scale  = np.sqrt(eigenvalues[k])
        beta_T = result_fpca["eigenfuncs"][k, 0]
        beta_S = result_fpca["eigenfuncs"][k, 1]
        xi_T   = beta_T @ Phi_T.T
        xi_S   = beta_S @ Phi_S.T

        for col, (mu, xi, label, unit) in enumerate([
            (mu_T, xi_T, "Température", "°C"),
            (mu_S, xi_S, "Salinité",    "psu"),
        ]):
            ax = axes[k, col]
            ax.plot(mu,            -depths, "k",   lw=2,   label="µ")
            ax.plot(mu + scale*xi, -depths, "r--", lw=1.5, label="µ + √λ·ξ")
            ax.plot(mu - scale*xi, -depths, "b--", lw=1.5, label="µ − √λ·ξ")
            ax.set_xlabel(f"{label} ({unit})")
            ax.set_ylabel("Profondeur (m)")
            ax.set_title(f"Mode {k+1} — {label}  (λ={eigenvalues[k]:.3f})")
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FPCA bivariée (T/S) + clustering GMM sur coefficients B-spline."
    )
    parser.add_argument(
        "path_in",
        type=str,
        help="Chemin vers le fichier NetCDF d'entrée.",
    )
    parser.add_argument(
        "path_out_nc",
        type=str,
        help="Chemin de sortie pour le NetCDF enrichi (ex: output/result_fpca_gmm.nc).",
    )
    parser.add_argument(
        "path_out_figs",
        type=str,
        help="Dossier de sortie pour les figures (ex: figures/).",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Nombre de clusters GMM (optionnel — sélection automatique par BIC si absent).",
    )
    parser.add_argument(
        "--var_threshold",
        type=float,
        default=0.90,
        help="Seuil de variance cumulée pour sélectionner les composantes FPCA (défaut: 0.90).",
    )
    args = parser.parse_args()

    os.makedirs(args.path_out_figs, exist_ok=True)

    def save_fig(fig, name):
        path = os.path.join(args.path_out_figs, name)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Figure sauvegardée : {path}")
        plt.close(fig)

    print("=== 1. FPCA bivariée ===")
    result_fpca = fpca_bivariate_bspline(args.path_in)

    print("\n=== 2. Clustering GMM ===")
    result_gmm = gmm_clustering_fpca(
        result_fpca,
        var_threshold=args.var_threshold,
        n_clusters=args.n_clusters,
    )

    print("\n=== 3. Sauvegarde NetCDF ===")
    save_results_netcdf(args.path_in, result_fpca, result_gmm, args.path_out_nc)

    print("\n=== 4. Figures ===")

    if result_gmm["bics"] is not None:
        k_range = range(2, 2 + len(result_gmm["bics"]))
        fig_bic, ax_bic = plt.subplots(figsize=(6, 3))
        ax_bic.plot(list(k_range), result_gmm["bics"], "o-", color="steelblue")
        ax_bic.axvline(result_gmm["n_clusters"], color="coral", ls="--",
                       label=f"optimal k={result_gmm['n_clusters']}")
        ax_bic.set_xlabel("Nombre de clusters")
        ax_bic.set_ylabel("BIC")
        ax_bic.set_title("Sélection du nombre de clusters")
        ax_bic.legend()
        plt.tight_layout()
        save_fig(fig_bic, "bic.png")

    fig_eig, _ = plt.subplots()
    plt.close(fig_eig)
    plot_eigenfunctions(result_fpca, n_modes=3)
    fig_eig = plt.gcf()
    save_fig(fig_eig, "eigenfunctions.png")

    plot_cluster_map(result_fpca, result_gmm, args.path_in)
    fig_map = plt.gcf()
    save_fig(fig_map, "cluster_map.png")