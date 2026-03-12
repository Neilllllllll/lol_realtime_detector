# utf-8
import socket

class SocketClient:
    def __init__(self, host='localhost', port=5566):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = (host, port)

    def send_message(self, message : bytes):
        self.socket.sendto(message, self.server_address)

    def receive_message(self, buffer_size=1024):
        return self.socket.recvfrom(buffer_size)[0]