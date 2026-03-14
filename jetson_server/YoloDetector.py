import os
import sys
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
        self.device = self.get_device()
        self.model = self.load_model(model_path)
        self.min_thresh = min_thresh

    def get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = YOLO(model_path, task="detect")
        model.to(self.device)

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
        except Exception as e:
            raise RuntimeError(f"Error during model warmup: {e}")
