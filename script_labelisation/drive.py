import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from config import Config

class DriveUploader:
    def __init__(self, id_dossier_drive: str, file_identifiants: str):
        self.id_dossier_drive = id_dossier_drive
        self.file_identifiants = file_identifiants
        self.con_drive = self.authenticate()

    def authenticate(self):
        """Gère l'authentification avec Google Drive."""
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile(self.file_identifiants)

        if gauth.credentials is None:
            print("Première connexion : Ouverture du navigateur...")
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            print("Connexion expirée : Renouvellement automatique...")
            gauth.Refresh()
        else:
            print("Connexion automatique réussie !")
            gauth.Authorize()

        gauth.SaveCredentialsFile(self.file_identifiants)
        return GoogleDrive(gauth)

    def create_drive_folder(self, folder_name: str, parent_id: str = None) -> str:
        """Crée un dossier sur Google Drive et retourne son ID."""
        metadata = {
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }

        if parent_id:
            metadata['parents'] = [{'id': parent_id}]

        folder = self.con_drive.CreateFile(metadata)
        folder.Upload()
        return folder['id']

    def upload_file(self, file_path: str, parent_id: str):
        """Upload un fichier dans un dossier Drive donné."""
        file_drive = self.con_drive.CreateFile({
            'title': os.path.basename(file_path),
            'parents': [{'id': parent_id}]
        })
        file_drive.SetContentFile(file_path)
        file_drive.Upload()
        print(f"Fichier uploadé : {file_path}")

    def upload_folder(self, folder_path: str, parent_id: str = None):
        """
        Upload récursivement un dossier local vers Google Drive.
        Retourne l'ID du dossier créé sur Drive.
        """
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Ce chemin n'est pas un dossier : {folder_path}")

        if parent_id is None:
            parent_id = self.id_dossier_drive

        folder_name = os.path.basename(os.path.normpath(folder_path))
        drive_folder_id = self.create_drive_folder(folder_name, parent_id)
        print(f"Dossier créé sur Drive : {folder_name} (ID: {drive_folder_id})")

        for item in os.listdir(folder_path):
            local_item_path = os.path.join(folder_path, item)

            if os.path.isdir(local_item_path):
                self.upload_folder(local_item_path, drive_folder_id)
            else:
                self.upload_file(local_item_path, drive_folder_id)

        return drive_folder_id

if __name__ == "__main__":
    ID_DOSSIER_DRIVE = Config.ID_DOSSIER_DRIVE
    FILE_IDENTIFIANTS = "mes_identifiants.txt"
    LOCAL_FOLDER = Config.LOCAL_FOLDER

    uploader = DriveUploader(
        id_dossier_drive=ID_DOSSIER_DRIVE,
        file_identifiants=FILE_IDENTIFIANTS
    )

    uploaded_folder_id = uploader.upload_folder(LOCAL_FOLDER)
    print(f"Dossier uploadé avec succès sur Google Drive ! ID : {uploaded_folder_id}")