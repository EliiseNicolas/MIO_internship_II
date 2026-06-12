import os
from pathlib import Path
from dotenv import load_dotenv
import copernicusmarine

# Charger le .env
load_dotenv()


def download_bathymetry(output_dir: str = "../../data/raw/bathymetry") -> Path:
    """
    Télécharge la bathymétrie statique CMEMS.
    Nécessite un compte Copernicus Marine Service.

    Args:
        output_dir: dossier de destination du fichier téléchargé

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

    print(f"Téléchargement de la bathymétrie dans : {output_path.resolve()}")

    copernicusmarine.get(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_static",
        output_directory=str(output_path),
        username=username,
        password=password,
        force_download=True,
    )

    print("Téléchargement terminé.")
    return output_path


if __name__ == "__main__":
    download_bathymetry()