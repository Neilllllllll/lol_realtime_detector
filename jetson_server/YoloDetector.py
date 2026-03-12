"""
Responsabilités :
- Charger le modèle YOLO
- Warmup du modèle pour réduire le temps de la première inférence
- Effectuer l'inférence pour détecter les objets dans une image
- Conversion des résultats en objet DetectionMessage
"""

import os
import sys

from ultralytics import YOLO

# Class for YOLO object detection
class YoloDetector:
    def __init__(self, model_path, min_thresh):
        self.model = self.load_model(model_path)
        self.min_thresh = min_thresh

    def load_model(self, model_path):
        if (not os.path.exists(model_path)):
            raise FileNotFoundError(f'Model file not found at {model_path}')
        return YOLO(model_path, task='detect')

    def detect_objects(self, image):
        results = self.model(image)
        return results[0].boxes
    
    def warmup_model(self):
        pass

    def build_detection_message(self):
        pass