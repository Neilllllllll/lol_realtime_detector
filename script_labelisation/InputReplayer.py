import argparse
import json
import time
from pathlib import Path
from pynput import keyboard, mouse

class InputReplayer:
    def __init__(self):
        self.mouse_controller = mouse.Controller()
        self.keyboard_controller = keyboard.Controller()

    @staticmethod
    def deserialize_key(key_data):
        kind = key_data["kind"]
        value = key_data["value"]

        if kind == "char":
            return value

        # Exemple: "Key.enter" -> keyboard.Key.enter
        if kind == "special" and value.startswith("Key."):
            key_name = value.split(".", 1)[1]
            return getattr(keyboard.Key, key_name)

        raise ValueError(f"Format de touche inconnu: {key_data}")

    @staticmethod
    def deserialize_button(button_str):
        # Exemple: "Button.left" -> mouse.Button.left
        if button_str.startswith("Button."):
            button_name = button_str.split(".", 1)[1]
            return getattr(mouse.Button, button_name)

        raise ValueError(f"Format de bouton inconnu: {button_str}")

    def replay_event(self, event):
        event_type = event["type"]
        data = event["data"]

        if event_type == "mouse_move":
            self.mouse_controller.position = (data["x"], data["y"])

        elif event_type == "mouse_click":
            self.mouse_controller.position = (data["x"], data["y"])
            button = self.deserialize_button(data["button"])
            if data["pressed"]:
                self.mouse_controller.press(button)
            else:
                self.mouse_controller.release(button)

        elif event_type == "mouse_scroll":
            self.mouse_controller.position = (data["x"], data["y"])
            self.mouse_controller.scroll(data["dx"], data["dy"])

        elif event_type == "key_press":
            key = self.deserialize_key(data["key"])
            self.keyboard_controller.press(key)

        elif event_type == "key_release":
            key = self.deserialize_key(data["key"])
            self.keyboard_controller.release(key)

        else:
            print(f"Événement ignoré: {event_type}")

    def run_single_file(self, input_file: str):
        payload = json.loads(Path(input_file).read_text(encoding="utf-8"))
        events = payload.get("events", [])

        if not events:
            print("Aucun événement à rejouer.")
            return

        print("Rejeu dans 3 secondes...")
        time.sleep(3)
        print("Rejeu démarré.")

        replay_start = time.perf_counter()

        for event in events:
            target_time = event["time"]

            while True:
                elapsed = time.perf_counter() - replay_start
                remaining = target_time - elapsed
                if remaining <= 0:
                    break
                # Petite pause pour éviter de monopoliser le CPU
                time.sleep(min(remaining, 0.001))

            try:
                self.replay_event(event)
            except Exception as exc:
                print(f"Erreur lors du rejeu de {event}: {exc}")

        print("Rejeu terminé.")
    
    def run_folder(self, folder: str):
        folder_path = Path(folder)
        if not folder_path.is_dir():
            print(f"Le chemin spécifié n'est pas un dossier valide: {folder}")
            return

        event_files = list(folder_path.glob("*.json"))
        if not event_files:
            print(f"Aucun fichier d'événements trouvé dans le dossier: {folder}")
            return

        for event_file in event_files:
            print(f"\nRejeu du fichier: {event_file}")
            self.run_single_file(str(event_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rejoueur d'événements clavier et souris")
    parser.add_argument("--folder", "-f", required=True, help="Dossier contenant les fichiers d'événements à rejouer")

    args = parser.parse_args()
    replayer = InputReplayer()

    replayer.run_folder(args.folder)