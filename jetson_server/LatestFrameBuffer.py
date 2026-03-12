"""
Responsabilités :
- le stockage thread-safe de la frame la plus récente    
- le remplacement de la frame précédente
- l’attente/réveil entre les threads
"""
import threading

# Represente une frame reçue du client
class FramePacket:
    def __init__(self, frame_id: int, timestamp: float = 3.1, frame_data: str = "lol"):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.frame_data = frame_data

class LatestFrameBuffer:
    def __init__(self):
        self.last_frame = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def update_frame(self,  frame: FramePacket):
        with self.lock:
            self.last_frame = frame
            self.condition.notify()

    def get_latest_frame(self):
        with self.lock:
            while self.last_frame is None:
                self.condition.wait()

            packet = self.last_frame
            self.last_frame = None
            return packet