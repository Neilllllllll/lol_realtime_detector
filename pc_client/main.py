import os
from pc_client.SocketClient import SocketClient
import time
from shared.logs.logs import Logger
import cv2

host, port = ('127.0.0.1', 5596)
socket = SocketClient(host, port)
logger = Logger()

images_folder = "pc_client/images_test"

try:
    for image in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image)
        frame = cv2.imread(image_path)
        print("taille de l'image en octets : ", os.path.getsize(image_path))
        img_resized = cv2.resize(frame, (640, 480))
        cv2.imshow("Image", img_resized)
except ConnectionRefusedError:
    logger.error("Connexion au serveur échoué ! ")