import os

from input_replayer import InputReplayer
from drive import DriveUploader
from config import Config
import json
import time

if __name__ == "__main__":
    folder_screenshots = "champ_folder_temp"
    # 1. Créer un dossier où on stockera temporairement les images des champions
    os.makedirs(folder_screenshots, exist_ok=True)

    # 2. Initialiser la connexion à Google Drive
    drive_con = DriveUploader(
        id_dossier_drive=Config.ID_DOSSIER_DRIVE,
        file_identifiants="mes_identifiants.txt"
    )

    # 3. Charger la liste des champions depuis le fichier JSON
    with open("../champions.json") as f:
        data = json.load(f)
    
    for champion in data["champions"]:
        # 4. Créer un folder pour stocker les images du champion
        os.makedirs(f"champ_folder/{champion}", exist_ok=True)

        input_replayer = InputReplayer(champion_name=champion, folder_screenshots=f"{folder_screenshots}/{champion}")
        # 5. On pointe directement vers le dossier 'actions' qui est juste à côté
        input_replayer.run_single_file("actions/launch_game.json")
        time.sleep(60) # 1 minute d'attente
        
        input_replayer.run_single_file("actions/play_and_leave.json")
        time.sleep(20) # 1 minute d'attente

        # 6. On upload le dossier du champion sur Google Drive
        drive_con.upload_folder(f"champ_folder/{champion}")
        
        os.rmdir(f"champ_folder/{champion}")