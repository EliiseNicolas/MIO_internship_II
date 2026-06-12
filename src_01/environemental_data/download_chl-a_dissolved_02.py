import os
from pathlib import Path
from dotenv import load_dotenv
import copernicusmarine

# Charger le .env
load_dotenv()


def download_o2_chl(
    date_start: str,
    date_end: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    output_dir: str = "../../data/raw/o2_chl",
) -> Path:
    """
    Télécharge l'oxygène dissous (O2) et la concentration en chlorophylle-a (chl)
    journaliers depuis CMEMS.
    Nécessite un compte Copernicus Marine Service.

    Datasets utilisés :
      - cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m  → o2  (µmol/kg)
      - cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m  → chl (mg/m³)

    Args:
        date_start  : date de début au format 'YYYY-MM-DD'
        date_end    : date de fin   au format 'YYYY-MM-DD'
        lat_min     : latitude minimale  (ex: 30.0)
        lat_max     : latitude maximale  (ex: 46.0)
        lon_min     : longitude minimale (ex: -6.0)
        lon_max     : longitude maximale (ex: 37.0)
        output_dir  : dossier de destination des fichiers téléchargés

    Returns:
        Path: chemin vers le dossier de téléchargement
    """
    # Lire les credentials
    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")

    # Vérifier que les credentials sont bien définis
    if not username or not password:
        raise ValueError(
            "Identifiants Copernicus manquants.\n"
            "1. Copie .env.example en .env\n"
            "2. Remplis COPERNICUS_USERNAME et COPERNICUS_PASSWORD"
        )

    # Créer le dossier de destination
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Paramètres du téléchargement :")
    print(f"  Période   : {date_start} → {date_end}")
    print(f"  Latitude  : {lat_min}° N → {lat_max}° N")
    print(f"  Longitude : {lon_min}° E → {lon_max}° E")
    print(f"  Dossier   : {output_path.resolve()}")
    print()

    common_kwargs = dict(
        start_datetime=f"{date_start}T00:00:00",
        end_datetime=f"{date_end}T23:59:59",
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        output_directory=str(output_path),
        username=username,
        password=password,
        force_download=True,
    )

    # ── 1. Oxygène dissous ────────────────────────────────────────────────────
    # Source : Global Ocean Biogeochemistry Analysis and Forecast
    # Dataset : Primary production and O2, daily (0.25°, analyse + prévision)
    # Variable: o2 — concentration en oxygène dissous (µmol/kg)
    print("─" * 60)
    print("[1/2] Téléchargement de l'oxygène dissous (o2)…")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
        variables=["o2"],
        **common_kwargs,
    )
    print("  → O2 téléchargé.")

    # ── 2. Chlorophylle-a ─────────────────────────────────────────────────────
    # Source : Global Ocean Biogeochemistry Analysis and Forecast
    # Dataset : Phytoplankton, daily (0.25°, analyse + prévision)
    # Variable: chl — mass concentration of chlorophyll a in sea water (mg/m³)
    print()
    print("─" * 60)
    print("[2/2] Téléchargement de la chlorophylle-a (chl)…")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
        variables=["chl"],
        **common_kwargs,
    )
    print("  → Chl-a téléchargée.")

    print()
    print("=" * 60)
    print("Téléchargement terminé.")
    print(f"Fichiers disponibles dans : {output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Télécharge l'oxygène dissous (o2) et la chlorophylle-a (chl) "
            "journaliers depuis CMEMS."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--date-start", required=True,  help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--date-end",   required=True,  help="Date de fin   (YYYY-MM-DD)")
    parser.add_argument("--lat-min",    required=True,  type=float, help="Latitude minimale  (degrés N)")
    parser.add_argument("--lat-max",    required=True,  type=float, help="Latitude maximale  (degrés N)")
    parser.add_argument("--lon-min",    required=True,  type=float, help="Longitude minimale (degrés E)")
    parser.add_argument("--lon-max",    required=True,  type=float, help="Longitude maximale (degrés E)")
    parser.add_argument(
        "--output-dir",
        default="../../data/raw/o2_chl",
        help="Dossier de destination",
    )

    args = parser.parse_args()

    download_o2_chl(
        date_start=args.date_start,
        date_end=args.date_end,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        output_dir=args.output_dir,
    )