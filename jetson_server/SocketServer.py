"""
Responsabilité :
- Gère la session client 
- Le thread reception
- Le thread d'inférence
- L'envoi des messages de détection au client
- le cycle de vie du serveur
"""
from shared.protocol.detection_schema import DetectionMessage
import socket 

class SocketServer:
    def __init__(self, host='', port=5596):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.client_address = None
        self.running = False
    
    def receive_frame(self):
        data, addr = self.socket.recvfrom(65536)
        self.client_address = addr
        print(f"Message reçu de {addr} : {len(data)} bytes")
        return data
    
    def show_status(self):
        return f"Client address: {self.client_address}, Running: {self.running}"
    
    def is_running(self):
        return self.running
    
    def start(self):
        self.running = True

    def end(self):
        self.running = False
        if self.socket:
            self.socket.close()

    def send_detection_message(self, detection_message: DetectionMessage):
        self.socket.sendto(detection_message.to_json().encode(), self.client_address)