import os

from dotenv import load_dotenv

load_dotenv("jetson_server/env/.env") 

model_path = os.getenv('MODEL_PATH', 'jetson_server/models/runs/detect/train2/weights/best.pt')
min_thresh = float(os.getenv('MIN_THRESH', 0.5))
hostname = os.getenv('HOSTNAME', 'localhost')
port = int(os.getenv('PORT', 5596))
