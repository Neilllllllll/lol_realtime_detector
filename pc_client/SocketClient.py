# utf-8
import socket

class SocketClient:
    def __init__(self, host: str = "localhost", port: int = 5596):
        self.host = host
        self.port = port
        self.server_address = (host, port)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False

    def connect(self):
        self.socket.connect(self.server_address)
        self.connected = True

    def _recv_exact(self, size: int) -> bytes | None:
        """
        Lit exactement 'size' octets depuis le socket.
        Retourne None si la connexion est fermée.
        """
        data = b""
        while len(data) < size:
            chunk = self.socket.recv(size - len(data))
            if not chunk:
                self.connected = False
                return None
            data += chunk
        return data

    def send_message(self, message: bytes):
        """
        Envoie un message binaire avec préfixe de taille sur 4 octets.
        """
        if not self.connected:
            raise RuntimeError("Le client n'est pas connecté au serveur.")

        header = len(message).to_bytes(4, byteorder="big")
        self.socket.sendall(header + message)

    def receive_message(self) -> bytes | None:
        """
        Reçoit un message binaire avec préfixe de taille sur 4 octets.
        """
        if not self.connected:
            raise RuntimeError("Le client n'est pas connecté au serveur.")

        header = self._recv_exact(4)
        if header is None:
            return None

        message_size = int.from_bytes(header, byteorder="big")
        if message_size <= 0:
            return None

        payload = self._recv_exact(message_size)
        return payload

    def close(self):
        self.connected = False
        if self.socket:
            self.socket.close()