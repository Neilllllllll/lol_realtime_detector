import os

model_path = os.getenv('MODEL_PATH', 'jetson_server/models/runs/detect/train2/weights/best.pt')
min_thresh = float(os.getenv('MIN_THRESH', 0.5))
