import os
from pathlib import Path
from datetime import date
from dotenv import load_dotenv, find_dotenv
import copernicusmarine

# Charger le .env
load_dotenv(find_dotenv())   # cherche dans les dossiers parents automatiquement


def download_temp_salinity(
    date_start: str,
    date_end: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    output_dir: str = "../../data/raw/temp_salinity",
) -> Path:
    """
    Télécharge la température et la salinité journalières depuis CMEMS.
    Nécessite un compte Copernicus Marine Service.

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
    print(f"Répertoire courant : {os.getcwd()}")
    print(f"Fichier .env trouvé : {find_dotenv()}")
    print(f"USERNAME lu : {os.getenv('COPERNICUS_USERNAME')}")
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

    print(f"Paramètres du téléchargement :")
    print(f"  Période   : {date_start} → {date_end}")
    print(f"  Latitude  : {lat_min}° N → {lat_max}° N")
    print(f"  Longitude : {lon_min}° E → {lon_max}° E")
    print(f"  Dossier   : {output_path.resolve()}")
    print()

    # Dataset journalier CMEMS — température (thetao) et salinité (so)
    # Source : Global Ocean Physics Reanalysis GLORYS12V1 (~8 km, 1/12°, 50 niveaux)
    # Couverture : Global (dont Océan Austral), 1993 → présent
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao", "so"],          # température et salinité
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

    print("\nTéléchargement terminé.")
    print(f"Fichiers disponibles dans : {output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Télécharge la température et la salinité journalières depuis CMEMS.",
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
        default="../../data/raw/temp_salinity",
        help="Dossier de destination",
    )

    args = parser.parse_args()

    download_temp_salinity(
        date_start=args.date_start,
        date_end=args.date_end,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        output_dir=args.output_dir,
    )
