import os
from dotenv import load_dotenv

load_dotenv("jetson_server/env/.env") 

models_path = os.getenv('MODELS_PATH', 'jetson_server/models/')
model_name = os.getenv('MODEL_NAME', 'custom_yolov11n.pt')
min_thresh = float(os.getenv('MIN_THRESH', 0.5))
hostname = os.getenv('HOSTNAME', 'localhost')
port = int(os.getenv('PORT', 5596))
