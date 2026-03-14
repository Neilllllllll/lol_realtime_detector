import os
from dotenv import load_dotenv

load_dotenv("pc_client/env/.env") 

hostname = os.getenv('HOSTNAME', 'localhost')
port = int(os.getenv('PORT', 5596))
