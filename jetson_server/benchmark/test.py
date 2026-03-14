import os
import time
from pathlib import Path
from jetson_server.conf.config import model_path, min_thresh
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from shared.logs.logs import Logger
from jetson_server.YoloDetector import YoloDetector

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

def get_model_names(model_dir: str) -> list[Path]:
    folder = Path(model_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Le dossier de modèles n'existe pas : {model_dir}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Le chemin n'est pas un dossier : {model_dir}")

    model_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".pt", ".pth"}]
    model_files.sort()
    return model_files

def benchmark_inference(detector: YoloDetector, image_files: list[Path], logger: Logger) -> None:
    if not image_files:
        logger.error("Aucune image trouvée dans le dossier.")
        return

    logger.info(f"Nombre d'images trouvées : {len(image_files)}")

    # Warmup avec la première image
    logger.info(f"Warmup avec : {image_files[0]}")
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

        logger.success(
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

    logger.info("\n========== RÉSULTATS ==========")
    logger.info(f"Device utilisé          : {detector.device}")
    logger.info(f"Nombre d'images         : {len(image_files)}")
    logger.info(f"Temps total             : {total_time:.4f} s")
    logger.info(f"Temps moyen / image     : {avg_time * 1000:.2f} ms")
    logger.info(f"Temps min / image       : {min_time * 1000:.2f} ms")
    logger.info(f"Temps max / image       : {max_time * 1000:.2f} ms")
    logger.info(f"FPS moyen               : {fps:.2f}")
    logger.info(f"Nb total de détections  : {total_detections}")

    if torch.cuda.is_available():
        logger.info(f"Mémoire GPU allouée     : {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"Mémoire GPU réservée    : {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    MODEL_PATH = model_path
    IMAGES_DIR = "jetson_server/tests/images_test"
    MIN_THRESH = min_thresh

    logger = Logger()
    logger.info(f"Version torch : {torch.__version__}")
    logger.info(f"CUDA disponible : {torch.cuda.is_available()}")

    image_files = get_image_files(IMAGES_DIR)
    models_path_list = get_model_names(MODEL_PATH)
    for i, model_path in enumerate(models_path_list, start=1):
        logger.info(f"Modèle trouvé : {model_path} lancement du benchmark... numéro {i}")
        detector = YoloDetector(model_path=model_path, min_thresh=MIN_THRESH)
        benchmark_inference(detector, image_files, logger)
        unloaded = torch.cuda.empty_cache()