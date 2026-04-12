import os
import json
import time
import threading

from input_replayer import InputReplayer
from lol_realtime_detector.script_labelisation.DriveUploader import DriveUploader
from config import Config

def initialiser_drive_avec_validation(folder_actions: str, file_identifiants: str = "mes_identifiants.txt") -> DriveUploader:
    """
    Lance l'initialisation de DriveUploader dans un thread pour éviter
    que le script principal soit bloqué pendant l'attente de validation auth.
    Ensuite, exécute automatiquement le fichier valide_auth.json.
    """
    result = {"drive_con": None, "exception": None}

    def init_drive():
        try:
            result["drive_con"] = DriveUploader(
                id_dossier_drive=Config.ID_DOSSIER_DRIVE,
                file_identifiants=file_identifiants
            )
        except Exception as e:
            result["exception"] = e

    # Lancement de l'initialisation bloquante dans un thread
    thread = threading.Thread(target=init_drive, daemon=True)
    thread.start()

    # Petit délai pour laisser le temps à la demande d'auth de s'afficher
    time.sleep(2)

    # Lance la validation automatique pendant que DriveUploader attend
    input_replayer = InputReplayer(champion_name=None, folder_screenshots=None)
    input_replayer.run_single_file(f"{folder_actions}/valide_auth.json")

    # On attend que l'initialisation se termine
    thread.join()

    if result["exception"] is not None:
        raise result["exception"]

    if result["drive_con"] is None:
        raise RuntimeError("Impossible d'initialiser la connexion Google Drive.")

    return result["drive_con"]


def supprimer_contenu_dossier(path: str) -> None:
    if not os.path.exists(path):
        return

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    os.rmdir(path)


if __name__ == "__main__":
    folder_actions = "actions"
    folder_screenshots = "champion_screenshots"
    file_identifiants = "mes_identifiants.txt"

    # Initialisation Google Drive + validation auth automatique
    drive_con = initialiser_drive_avec_validation(
        folder_actions=folder_actions,
        file_identifiants=file_identifiants
    )

    # Chargement de la liste des champions
    with open("../champions.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    compteur_reinit = 0

    for champion in data["champions"]:
        compteur_reinit += 1

        if compteur_reinit >= 2:
            print(f"Réinitialisation de la connexion à Google Drive (itération {compteur_reinit})...")

            if os.path.exists(file_identifiants):
                os.remove(file_identifiants)

            drive_con = initialiser_drive_avec_validation(
                folder_actions=folder_actions,
                file_identifiants=file_identifiants
            )
            compteur_reinit = 0

        champion_folder = os.path.join(folder_screenshots, champion)

        # Si le dossier existe déjà, on l'upload puis on le supprime
        if os.path.exists(champion_folder):
            drive_con.upload_folder(champion_folder)
            supprimer_contenu_dossier(champion_folder)
            continue

        os.makedirs(champion_folder, exist_ok=True)

        input_replayer = InputReplayer(
            champion_name=champion,
            folder_screenshots=champion_folder
        )

        input_replayer.run_single_file(f"{folder_actions}/launch_game.json")
        time.sleep(30)

        input_replayer.run_single_file(f"{folder_actions}/play_and_leave.json")
        time.sleep(20)

        # Upload du dossier du champion sur Google Drive
        drive_con.upload_folder(champion_folder)

        # Suppression du dossier local
        supprimer_contenu_dossier(champion_folder)