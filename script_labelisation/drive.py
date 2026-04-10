import json
import os
import glob
import time
import gc
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# --- CONFIGURATION ---
# Le 'r' avant les guillemets est obligatoire pour les chemins Windows
PATH_LOCAL = r"C:\Users\valen\Videos\NVIDIA\League of Legends\*.mp4"
ID_DOSSIER_DRIVE = "1xogPi4q2TkGkvDwBgDxuEG2aKWY9ybQE" # Remplacez par l'ID de votre dossier sur Google Drive
FILE_IDENTIFIANTS = "mes_identifiants.txt"

def get_latest_video(path):
    """Trouve la vidéo la plus récente dans le dossier spécifié."""
    files = glob.glob(path)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def get_or_create_folder(drive, folder_name):
    """Trouve le dossier sur le Drive, ou le crée s'il n'existe pas."""
    query = f"title = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    file_list = drive.ListFile({'q': query}).GetList()
    
    if file_list:
        return file_list[0]['id']
    else:
        folder = drive.CreateFile({'title': folder_name, 'mimeType': 'application/vnd.google-apps.folder'})
        folder.Upload()
        return folder['id']

def main(nom_video):
    # 1. AUTHENTIFICATION AVEC SAUVEGARDE
    gauth = GoogleAuth()
    
    # Charge les identifiants s'ils existent
    gauth.LoadCredentialsFile(FILE_IDENTIFIANTS)
    
    if gauth.credentials is None:
        print("Première connexion : Ouverture du navigateur...")
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        print("Connexion expirée : Renouvellement automatique...")
        gauth.Refresh()
    else:
        print("Connexion automatique réussie !")
        gauth.Authorize()
    
    # Sauvegarde pour la prochaine fois
    gauth.SaveCredentialsFile(FILE_IDENTIFIANTS)

    drive = GoogleDrive(gauth)

    # 2. RECHERCHE DE LA VIDÉO
    video_path = get_latest_video(PATH_LOCAL)
    
    if video_path:
        nouveau_nom = os.path.basename(video_path) # Garde le nom d'origine de la vidéo
        print(f"Fichier trouvé : {nouveau_nom}")

        # 4. UPLOAD
        file_drive = drive.CreateFile({
            'title': nom_video,
            'parents': [{'id': ID_DOSSIER_DRIVE}]
        })

        print("Envoi en cours sur Google Drive...")
        file_drive.SetContentFile(video_path)
        file_drive.Upload()
        print("Upload terminé avec succès !")

        # 5. NETTOYAGE ET SUPPRESSION FORCÉE
        print("Nettoyage de la mémoire et attente de libération du fichier...")
        
        # On force PyDrive et Python à tout lâcher
        file_drive = None 
        gc.collect()      
        
        # Boucle pour patienter si Windows ou NVIDIA bloque encore le fichier
        supprime = False
        for i in range(10): # 10 essais maximum
            try:
                os.remove(video_path)
                print(f"✅ Fichier local supprimé avec succès : {nouveau_nom}")
                supprime = True
                break
            except PermissionError:
                print(f"⏳ Tentative {i+1}/10 : Fichier bloqué (NVIDIA/Antivirus). Nouvel essai dans 3s...")
                time.sleep(3)
                
        if not supprime:
            print("❌ Impossible de supprimer la vidéo. Elle est définitivement verrouillée par un autre processus.")
    else:
        print("Aucune vidéo trouvée dans le dossier.")

if __name__ == "__main__":
    with open("../champions.json") as f:
        data = json.load(f)
    for champion in data["champions"]:
        main(champion)