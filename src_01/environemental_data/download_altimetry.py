import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import copernicusmarine

# Charger le .env
load_dotenv(find_dotenv())


def download_altimetry(
    date_start: str,
    date_end: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    output_dir: str = "../../data/raw/altimetry",
) -> Path:
    """
    Télécharge ADT, SLA et courants géostrophiques (u, v) depuis CMEMS.
    Nécessite un compte Copernicus Marine Service.

    Produit : SEALEVEL_GLO_PHY_L4_MY_008_047
              DUACS multimission altimetry — reprocessed (delayed time)
              Résolution : 1/4°, daily, global, 1993 → présent

    Variables téléchargées :
        adt    : Absolute Dynamic Topography          (m)
        sla    : Sea Level Anomaly                    (m)
        ugos   : Absolute geostrophic current (zonal) (m/s)
        vgos   : Absolute geostrophic current (merid) (m/s)
        ugosa  : Geostrophic current anomaly (zonal)  (m/s)
        vgosa  : Geostrophic current anomaly (merid)  (m/s)

    Args:
        date_start : date de début  'YYYY-MM-DD'
        date_end   : date de fin    'YYYY-MM-DD'
        lat_min/max: bornes latitude  (degrés N, ex: -60.0 / -20.0)
        lon_min/max: bornes longitude (degrés E, ex:  30.0 /  80.0)
        output_dir : dossier de destination

    Returns:
        Path : chemin vers le dossier de téléchargement
    """
    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")

    print(f"Répertoire courant : {os.getcwd()}")
    print(f"Fichier .env trouvé : {find_dotenv()}")
    print(f"USERNAME lu : {username}")

    if not username or not password:
        raise ValueError(
            "Identifiants Copernicus manquants.\n"
            "1. Copie .env.example en .env\n"
            "2. Remplis COPERNICUS_USERNAME et COPERNICUS_PASSWORD"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nParamètres du téléchargement :")
    print(f"  Produit   : SEALEVEL_GLO_PHY_L4_MY_008_047 (DUACS reprocessed)")
    print(f"  Variables : adt, sla, ugos, vgos, ugosa, vgosa")
    print(f"  Période   : {date_start} → {date_end}")
    print(f"  Latitude  : {lat_min}° → {lat_max}°")
    print(f"  Longitude : {lon_min}° → {lon_max}°")
    print(f"  Dossier   : {output_path.resolve()}\n")

    copernicusmarine.subset(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        variables=["adt", "sla", "ugos", "vgos", "ugosa", "vgosa"],
        start_datetime=f"{date_start}T00:00:00",
        end_datetime=f"{date_end}T23:59:59",
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        output_directory=str(output_path),
        username=username,
        password=password,
    )

    print("\nTéléchargement terminé.")
    print(f"Fichiers disponibles dans : {output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Télécharge ADT, SLA et courants géostrophiques depuis CMEMS.",
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
        default="../../data/raw/altimetry",
        help="Dossier de destination",
    )

    args = parser.parse_args()

    download_altimetry(
        date_start=args.date_start,
        date_end=args.date_end,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        output_dir=args.output_dir,
    )