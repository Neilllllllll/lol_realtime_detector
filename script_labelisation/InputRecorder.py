import json
import threading
import time
from pathlib import Path
from pynput import keyboard, mouse
import argparse
from screeninfo import get_monitors


class InputRecorder:
    def __init__(self, output_folder: str, file_name: str = "1_event.json"):
        self.events = []
        self.lock = threading.Lock()
        self.start_time = None
        self.running = True
        self.mouse_listener = None
        self.keyboard_listener = None
        self.output_folder = output_folder
        self.file_name = file_name
        
    def now(self) -> float:
        return time.perf_counter() - self.start_time

    def add_event(self, event_type: str, data: dict):
        with self.lock:
            self.events.append({
                "time": self.now(),
                "type": event_type,
                "data": data,
            })

    @staticmethod
    def serialize_key(key):
        if isinstance(key, keyboard.KeyCode):
            return {
                "kind": "char",
                "value": key.char
            }
        return {
            "kind": "special",
            "value": str(key)
        }

    def on_key_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.running = False
                # On n'enregistre pas ESC pour éviter de le rejouer par défaut
                return False

            self.add_event("key_press", {
                "key": self.serialize_key(key)
            })
        except Exception as exc:
            print(f"Erreur on_key_press: {exc}")

    def on_key_release(self, key):
        try:
            if key == keyboard.Key.esc:
                return False

            self.add_event("key_release", {
                "key": self.serialize_key(key)
            })
        except Exception as exc:
            print(f"Erreur on_key_release: {exc}")

    def on_move(self, x, y):
        try:
            self.add_event("mouse_move", {"x": x, "y": y})
        except Exception as exc:
            print(f"Erreur on_move: {exc}")

    def on_click(self, x, y, button, pressed):
        try:
            self.add_event("mouse_click", {
                "x": x,
                "y": y,
                "button": str(button),
                "pressed": pressed
            })
        except Exception as exc:
            print(f"Erreur on_click: {exc}")

    def center_mouse(self):
        monitor = get_monitors()[0]  # écran principal
        width = monitor.width
        height = monitor.height
        mouse.Controller().position = (width // 2, height // 2)

    def on_scroll(self, x, y, dx, dy):
        try:
            self.add_event("mouse_scroll", {
                "x": x,
                "y": y,
                "dx": dx,
                "dy": dy
            })
        except Exception as exc:
            print(f"Erreur on_scroll: {exc}")

    def save(self, path: str):
        payload = {
            "version": 1,
            "created_at": time.time(),
            "events": self.events
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_in_folder(self, file_name: str):
        output_path = Path(self.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        self.save(str(output_path / file_name))

    def run(self):
        self.start_time = time.perf_counter()

        self.center_mouse()
        print("Enregistrement en cours...")
        print("Appuie sur ESC pour arrêter et sauvegarder.\n")

        self.mouse_listener = mouse.Listener(
            # on_move=self.on_move,
            on_click=self.on_click,
            on_scroll=self.on_scroll
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )

        self.mouse_listener.start()
        self.keyboard_listener.start()

        self.keyboard_listener.join()
        self.mouse_listener.stop()

        self.save_in_folder(self.file_name)
        print(f"\nEnregistrement terminé. Fichier sauvegardé : {self.output_folder + self.file_name}")
        print(f"Nombre d'événements : {len(self.events)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enregistreur d'événements clavier et souris")
    parser.add_argument("--output_folder", "-o", required=True, help="Dossier de sortie pour les événements enregistrés")
    parser.add_argument("--file_name", "-n", default="1_event.json", help="Nom du fichier de sortie (par défaut: 1_event.json)")

    args = parser.parse_args()
    recorder = InputRecorder(args.output_folder, args.file_name)

    recorder.run()