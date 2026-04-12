import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pynput import keyboard, mouse
from screeninfo import get_monitors

"""
Class permettant d'enregistrer les événements clavier et souris, avec des fonctionnalités de contrôle de l'enregistrement (démarrage, insertion de marqueurs, capture d'écran) et de sauvegarde dans un format structuré.
"""

class InputRecorder:
    FORMAT_VERSION = 2

    def __init__(
        self,
        output_folder: str,
        file_name: str = "1_event.json",
        record_mouse_move: bool = False,
        start_key=keyboard.Key.f7,
        text_trigger_key=keyboard.Key.f8,
        screenshot_key=keyboard.Key.f9
    ):
        self.events: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

        self.start_time: Optional[float] = None
        self.running = True
        self.recording_started = False

        self.mouse_listener = None
        self.keyboard_listener = None

        self.output_folder = output_folder
        self.file_name = file_name
        self.record_mouse_move = record_mouse_move

        self.start_key = start_key
        self.text_trigger_key = text_trigger_key
        self.screenshot_key = screenshot_key
        
    def now(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time

    def add_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if not self.recording_started:
            return

        with self.lock:
            self.events.append({
                "time": self.now(),
                "type": event_type,
                "data": data,
            })

    @staticmethod
    def serialize_key(key: Any) -> Dict[str, Any]:
        if isinstance(key, keyboard.KeyCode):
            return {
                "kind": "char",
                "value": key.char
            }
        return {
            "kind": "special",
            "value": str(key)
        }

    def start_recording(self):
        self.center_mouse()
        self.start_time = time.perf_counter()
        self.recording_started = True

        print("\nEnregistrement démarré.")
        print(f"{self.text_trigger_key} : insérer un marqueur 'écrire le texte'")
        print("ESC : arrêter et sauvegarder")
        print(f"{self.screenshot_key} : prendre une capture d'écran")
        print()

    def on_key_press(self, key: Any):
        try:
            # ESC arrête toujours
            if key == keyboard.Key.esc:
                self.running = False
                return False

            # Tant que l'enregistrement n'a pas démarré, seule la touche start_key est acceptée
            if not self.recording_started:
                if key == self.start_key:
                    self.start_recording()
                return

            # Touche spéciale pour insérer un marqueur text_input
            if key == self.text_trigger_key:
                self.add_event("text_input", {})
                print(f"[Recorder] Marqueur 'écrire le texte' ajouté à t={self.now():.3f}s")
                return

            # Touche spéciale pour prendre une capture d'écran
            if key == self.screenshot_key:
                self.add_event("screenshot", {})
                print(f"[Recorder] Capture d'écran ajoutée à t={self.now():.3f}s")
                return

            # Ne pas enregistrer la touche de démarrage comme un input normal
            if key == self.start_key:
                return

            self.add_event("key_press", {
                "key": self.serialize_key(key)
            })

        except Exception as exc:
            print(f"Erreur on_key_press: {exc}")

    def on_key_release(self, key: Any):
        try:
            if key == keyboard.Key.esc:
                return False

            # Ignore toutes les touches tant que le record n'a pas démarré
            if not self.recording_started:
                return

            # Ne pas enregistrer les touches spéciales de contrôle
            if key in (self.start_key, self.text_trigger_key, self.screenshot_key):
                return

            self.add_event("key_release", {
                "key": self.serialize_key(key)
            })

        except Exception as exc:
            print(f"Erreur on_key_release: {exc}")

    def on_move(self, x: int, y: int):
        if not self.record_mouse_move or not self.recording_started:
            return

        try:
            self.add_event("mouse_move", {"x": x, "y": y})
        except Exception as exc:
            print(f"Erreur on_move: {exc}")

    def on_click(self, x: int, y: int, button: Any, pressed: bool):
        if not self.recording_started:
            return

        try:
            self.add_event("mouse_click", {
                "x": x,
                "y": y,
                "button": str(button),
                "pressed": pressed
            })
        except Exception as exc:
            print(f"Erreur on_click: {exc}")

    def on_scroll(self, x: int, y: int, dx: int, dy: int):
        if not self.recording_started:
            return

        try:
            self.add_event("mouse_scroll", {
                "x": x,
                "y": y,
                "dx": dx,
                "dy": dy
            })
        except Exception as exc:
            print(f"Erreur on_scroll: {exc}")

    def center_mouse(self):
        try:
            monitor = get_monitors()[0]
            width = monitor.width
            height = monitor.height
            mouse.Controller().position = (width // 2, height // 2)
        except Exception as exc:
            print(f"Impossible de centrer la souris: {exc}")

    def save(self, path: str):
        payload = {
            "version": self.FORMAT_VERSION,
            "created_at": time.time(),
            "record_mouse_move": self.record_mouse_move,
            "start_key": str(self.start_key),
            "text_trigger_key": str(self.text_trigger_key),
            "recording_started": self.recording_started,
            "events": self.events,
        }

        Path(path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def save_in_folder(self, file_name: str):
        output_path = Path(self.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        self.save(str(output_path / file_name))

    def run(self):
        print("Recorder prêt.")
        print(f"{self.start_key} : démarrer l'enregistrement")
        print(f"{self.text_trigger_key} : insérer un marqueur 'écrire le texte'")
        print(f"{self.screenshot_key} : prendre une capture d'écran")
        print("ESC : arrêter et sauvegarder")
        print()

        self.mouse_listener = mouse.Listener(
            on_move=self.on_move,
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

        try:
            self.mouse_listener.stop()
        except Exception:
            pass

        self.running = False
        self.save_in_folder(self.file_name)

        full_path = str(Path(self.output_folder) / self.file_name)
        print("\nEnregistrement terminé.")
        print(f"Fichier sauvegardé : {full_path}")
        print(f"Nombre d'événements : {len(self.events)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enregistreur d'événements clavier et souris")
    parser.add_argument(
        "--output_folder",
        "-o",
        required=True,
        help="Dossier de sortie pour les événements enregistrés"
    )
    parser.add_argument(
        "--file_name",
        "-n",
        default="1_event.json",
        help="Nom du fichier de sortie (par défaut: 1_event.json)"
    )
    parser.add_argument(
        "--record_mouse_move",
        action="store_true",
        help="Enregistrer aussi les mouvements de souris"
    )

    args = parser.parse_args()

    recorder = InputRecorder(
        output_folder=args.output_folder,
        file_name=args.file_name,
        record_mouse_move=args.record_mouse_move,
    )
    recorder.run()