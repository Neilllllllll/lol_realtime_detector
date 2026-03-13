import os
import time
from pathlib import Path
from jetson_server.conf.config import model_path, min_thresh
import torch
from ultralytics import YOLO

from shared.protocol.detection_schema import Detection


class YoloDetector:
    """
    Responsabilités :
    - Charger le modèle YOLO
    - Warmup du modèle pour réduire le temps de la première inférence
    - Effectuer l'inférence pour détecter les objets dans une image
    - Conversion des résultats en objet Detection
    """

    def __init__(self, model_path, min_thresh):
        self.device = self._get_device()
        print(f"[INFO] Device sélectionné : {self.device}")

        self.model = self.load_model(model_path)
        self.min_thresh = min_thresh

        if torch.cuda.is_available():
            print(f"[INFO] GPU : {torch.cuda.get_device_name(0)}")
            print(f"[INFO] CUDA version (torch) : {torch.version.cuda}")

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = YOLO(model_path, task="detect")
        model.to(self.device)

        try:
            print(f"[INFO] Device réel du modèle : {next(model.model.parameters()).device}")
        except Exception as e:
            print(f"[WARN] Impossible de lire le device réel du modèle : {e}")

        return model

    def detect_objects(self, image) -> list[Detection]:
        results = self.model.predict(
            source=image,
            device=0 if self.device.startswith("cuda") else "cpu",
            verbose=False
        )

        detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf >= self.min_thresh:
                    class_id = int(box.cls.item())
                    class_name = self.model.names[class_id]
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    width = x_max - x_min
                    height = y_max - y_min

                    detections.append(
                        Detection(
                            class_name=class_name,
                            confidence=conf,
                            bbox=[x_min, y_min, width, height],
                        )
                    )

        return detections

    def warmup(self, image_path):
        try:
            self.model.predict(
                source=image_path,
                device=0 if self.device.startswith("cuda") else "cpu",
                verbose=False
            )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            print(f"[INFO] Warmup effectué sur {self.device}")
        except Exception as e:
            print(f"[ERROR] Error during model warmup: {e}")


def is_image_file(path: Path) -> bool:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return path.suffix.lower() in valid_extensions


def get_image_files(images_dir: str) -> list[Path]:
    folder = Path(images_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Le dossier d'images n'existe pas : {images_dir}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Le chemin n'est pas un dossier : {images_dir}")

    image_files = [p for p in folder.iterdir() if p.is_file() and is_image_file(p)]
    image_files.sort()
    return image_files


def benchmark_inference(detector: YoloDetector, image_files: list[Path]) -> None:
    if not image_files:
        print("[ERROR] Aucune image trouvée dans le dossier.")
        return

    print(f"[INFO] Nombre d'images trouvées : {len(image_files)}")

    # Warmup avec la première image
    print(f"[INFO] Warmup avec : {image_files[0]}")
    detector.warmup(str(image_files[0]))

    timings = []
    total_detections = 0

    for idx, image_path in enumerate(image_files, start=1):
        start = time.perf_counter()

        detections = detector.detect_objects(str(image_path))

        if detector.device.startswith("cuda"):
            torch.cuda.synchronize()

        end = time.perf_counter()

        elapsed = end - start
        timings.append(elapsed)
        total_detections += len(detections)

        print(
            f"[{idx}/{len(image_files)}] "
            f"{image_path.name} -> "
            f"{elapsed * 1000:.2f} ms, "
            f"{len(detections)} détection(s)"
        )

    total_time = sum(timings)
    avg_time = total_time / len(timings)
    min_time = min(timings)
    max_time = max(timings)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    print("\n========== RÉSULTATS ==========")
    print(f"Device utilisé          : {detector.device}")
    print(f"Nombre d'images         : {len(image_files)}")
    print(f"Temps total             : {total_time:.4f} s")
    print(f"Temps moyen / image     : {avg_time * 1000:.2f} ms")
    print(f"Temps min / image       : {min_time * 1000:.2f} ms")
    print(f"Temps max / image       : {max_time * 1000:.2f} ms")
    print(f"FPS moyen               : {fps:.2f}")
    print(f"Nb total de détections  : {total_detections}")

    if torch.cuda.is_available():
        print(f"Mémoire GPU allouée     : {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Mémoire GPU réservée    : {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")


if __name__ == "__main__":
    # ============================
    # Paramètres à adapter
    # ============================
    MODEL_PATH = model_path
    IMAGES_DIR = "jetson_server/tests/images_test"
    MIN_THRESH = min_thresh

    print("[INFO] Version torch :", torch.__version__)
    print("[INFO] CUDA disponible :", torch.cuda.is_available())

    detector = YoloDetector(model_path=MODEL_PATH, min_thresh=MIN_THRESH)
    image_files = get_image_files(IMAGES_DIR)
    benchmark_inference(detector, image_files)
