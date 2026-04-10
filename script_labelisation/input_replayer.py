import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pynput import keyboard, mouse
import pyautogui


class InputReplayer:
    def __init__(self, folder_screenshots: str = "./screenshots", champion_name: str = "default"):
        self.mouse_controller = mouse.Controller()
        self.keyboard_controller = keyboard.Controller()
        self.folder_screenshots = folder_screenshots
        self.champion_name = champion_name

    @staticmethod
    def deserialize_key(key_data: Dict[str, Any]):
        kind = key_data["kind"]
        value = key_data["value"]

        if kind == "char":
            return value

        if kind == "special" and value.startswith("Key."):
            key_name = value.split(".", 1)[1]
            return getattr(keyboard.Key, key_name)

        raise ValueError(f"Format de touche inconnu: {key_data}")

    @staticmethod
    def deserialize_button(button_str: str):
        if button_str.startswith("Button."):
            button_name = button_str.split(".", 1)[1]
            return getattr(mouse.Button, button_name)

        raise ValueError(f"Format de bouton inconnu: {button_str}")

    def type_text(self, text: str):
        if not text:
            print("[Replayer] Chaîne vide fournie pour text_input.")
            return
        self.keyboard_controller.type(text)

    def replay_event(self, event: Dict[str, Any]):
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

        elif event_type == "text_input":
            self.type_text(self.champion_name)
        
        elif event_type == "screenshot":
            screenshot = pyautogui.screenshot()
            timestamp = int(time.time() * 1000)
            screenshot.save(f"{self.folder_screenshots}/{self.champion_name}_screenshot_{timestamp}.png")

        else:
            print(f"Événement ignoré: {event_type}")

    def load_events(self, input_file: str) -> List[Dict[str, Any]]:
        payload = json.loads(Path(input_file).read_text(encoding="utf-8"))
        return payload.get("events", [])

    def run_single_file(self, input_file: str):
        events = self.load_events(input_file)

        if not events:
            print(f"Aucun événement à rejouer dans {input_file}.")
            return

        print(f"Rejeu du fichier {input_file} dans 3 secondes...")
        time.sleep(3)
        print("Rejeu démarré.")

        replay_start = time.perf_counter()

        for event in events:
            target_time = event.get("time", 0)

            while True:
                elapsed = time.perf_counter() - replay_start
                remaining = target_time - elapsed
                if remaining <= 0:
                    break
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

        event_files = sorted(folder_path.glob("*.json"))
        if not event_files:
            print(f"Aucun fichier d'événements trouvé dans le dossier: {folder}")
            return

        for event_file in event_files:
            print(f"\n--- Rejeu du fichier: {event_file.name} ---")
            self.run_single_file(str(event_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rejoueur d'événements clavier et souris")
    parser.add_argument(
        "--folder",
        "-f",
        required=True,
        help="Dossier contenant les fichiers d'événements à rejouer"
    )
    parser.add_argument(
        "--champion",
        "-c",
        default="default",
        help="Nom du champion pour les captures d'écran"
    )
    parser.add_argument(
        "--screenshots_folder",
        "-s",
        default="./screenshots",
        help="Dossier où les captures d'écran seront sauvegardées"
    )
    
    args = parser.parse_args()
    replayer = InputReplayer(folder_screenshots=args.screenshots_folder, champion_name=args.champion, )
    replayer.run_folder(args.folder)