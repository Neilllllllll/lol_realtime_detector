import dotenv
dotenv.load_dotenv()

class Config:
    ID_DOSSIER_DRIVE: str = dotenv.get_key(dotenv.find_dotenv(), "ID_DOSSIER_DRIVE")
    LOCAL_FOLDER: str = dotenv.get_key(dotenv.find_dotenv(), "LOCAL_FOLDER")