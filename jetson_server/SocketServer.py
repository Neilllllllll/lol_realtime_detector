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
    def __init__(self, host: str = "", port: int = 5596, backlog: int = 1):
        self.host = host
        self.port = port
        self.backlog = backlog

        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.running = False

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.backlog)

        print(f"Serveur TCP en écoute sur {self.host or '0.0.0.0'}:{self.port}")

    def accept_client(self):
        if self.server_socket is None:
            raise RuntimeError("Le serveur n'est pas démarré. Appelle start() avant accept_client().")

        self.client_socket, self.client_address = self.server_socket.accept()
        self.running = True
        print(f"Client connecté depuis {self.client_address}")

    def show_status(self) -> str:
        return f"Client address: {self.client_address}, Running: {self.running}"

    def is_running(self) -> bool:
        return self.running

    def _recv_exact(self, size: int) -> bytes | None:
        """
        Lit exactement 'size' octets depuis le socket client.
        Retourne None si le client ferme la connexion.
        """
        if self.client_socket is None:
            raise RuntimeError("Aucun client connecté.")

        data = b""
        while len(data) < size:
            chunk = self.client_socket.recv(size - len(data))
            if not chunk:
                self.running = False
                return None
            data += chunk
        return data

    def receive_message(self) -> bytes | None:
        """
        Reçoit un message binaire encadré par une taille sur 4 octets.
        Format: [4 octets taille][payload]
        """
        header = self._recv_exact(4)
        if header is None:
            return None

        payload_size = int.from_bytes(header, byteorder="big")
        if payload_size <= 0:
            return None

        payload = self._recv_exact(payload_size)
        return payload

    def send_message(self, payload: bytes):
        """
        Envoie un message binaire encadré par une taille sur 4 octets.
        """
        if self.client_socket is None:
            raise RuntimeError("Aucun client connecté.")

        header = len(payload).to_bytes(4, byteorder="big")
        self.client_socket.sendall(header + payload)

    def send_detection_message(self, detection_message: DetectionMessage):
        payload = detection_message.to_dict().encode("utf-8")
        self.send_message(payload)

    def close_client(self):
        if self.client_socket is not None:
            try:
                self.client_socket.close()
            finally:
                self.client_socket = None
                self.client_address = None
                self.running = False

    def stop(self):
        self.running = False

        if self.client_socket is not None:
            try:
                self.client_socket.close()
            finally:
                self.client_socket = None

        if self.server_socket is not None:
            try:
                self.server_socket.close()
            finally:
                self.server_socket = None

        print("Serveur arrêté proprement.")