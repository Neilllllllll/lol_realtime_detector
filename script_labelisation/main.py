from input_replayer import InputReplayer
import json
import time

if __name__ == "__main__":
    # 1. On remonte d'un dossier (../) pour trouver le JSON à la racine
    print("Champions disponibles pour le test de labelisation :")
    with open("../champions.json") as f:
        data = json.load(f)
    
    for champion in data["champions"]:
        input_replayer = InputReplayer(champion)
        
        # 2. On pointe directement vers le dossier 'actions' qui est juste à côté
        input_replayer.run_single_file("actions/launch_game.json")
        time.sleep(60) # 1 minute d'attente
        
        input_replayer.run_single_file("actions/play_and_leave.json")
        time.sleep(20) # 1 minute d'attente