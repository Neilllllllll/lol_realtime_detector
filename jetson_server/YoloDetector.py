"""
Responsabilités :
- Charger le modèle YOLO
- Warmup du modèle pour réduire le temps de la première inférence
- Effectuer l'inférence pour détecter les objets dans une image
- Conversion des résultats en objet DetectionMessage
"""

import os
import sys

import torch

from ultralytics import YOLO
from shared.protocol.detection_schema import Detection

# Class for YOLO object detection
class YoloDetector:
    def __init__(self, model_path, min_thresh):
        self.model = self.load_model(model_path)
        self.min_thresh = min_thresh
        self.device = self.select_device()

    def select_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def load_model(self, model_path):
        if (not os.path.exists(model_path)):
            raise FileNotFoundError(f'Model file not found at {model_path}')
        return YOLO(model_path, task='detect')

    def detect_objects(self, image) -> list[Detection]:
        results = self.model(image)
        detections = []
        for result in results:
            for box in result.boxes:
                conf = box.conf.item()
                if conf >= self.min_thresh:
                    class_id = int(box.cls.item())
                    class_name = self.model.names[class_id]
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    width = x_max - x_min
                    height = y_max - y_min
                    detections.append(Detection(class_name=class_name, confidence=conf, bbox=[x_min, y_min, width, height]))
        return detections
    
    def warmup(self, image_path):
        try:
            self.model(image_path, warmup=True)
        except Exception as e:
            print(f"Error during model warmup: {e}")

    def print_device_info(self):
        if torch.cuda.is_available():
            print("GPU :", torch.cuda.get_device_name(0))
        print("Device du modèle :", next(self.model.model.parameters()).device)