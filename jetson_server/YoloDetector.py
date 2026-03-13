import os
import time
import torch

from ultralytics import YOLO
from shared.protocol.detection_schema import Detection


class YoloDetector:
    def __init__(self, model_path, min_thresh):
        self.device = self._get_device()
        print(f"[YoloDetector] selected device: {self.device}")

        self.model = self.load_model(model_path)
        self.min_thresh = min_thresh

        if torch.cuda.is_available():
            print(f"[YoloDetector] GPU name: {torch.cuda.get_device_name(0)}")
            print(f"[YoloDetector] memory allocated after load: {torch.cuda.memory_allocated(0)}")
            print(f"[YoloDetector] memory reserved after load: {torch.cuda.memory_reserved(0)}")

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
            print(f"[YoloDetector] actual model device: {next(model.model.parameters()).device}")
        except Exception as e:
            print(f"[YoloDetector] unable to inspect model device: {e}")

        return model

    def detect_objects(self, image) -> list[Detection]:
        start = time.time()

        results = self.model.predict(
            source=image,
            device=0 if self.device.startswith("cuda") else "cpu",
            verbose=False
        )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        end = time.time()
        print(f"[YoloDetector] inference time: {end - start:.4f}s")

        if torch.cuda.is_available():
            print(f"[YoloDetector] memory allocated: {torch.cuda.memory_allocated(0)}")
            print(f"[YoloDetector] memory reserved: {torch.cuda.memory_reserved(0)}")

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
                            bbox=[x_min, y_min, width, height]
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
            print(f"[YoloDetector] warmup done on {self.device}")
        except Exception as e:
            print(f"Error during model warmup on {self.device}: {e}")
