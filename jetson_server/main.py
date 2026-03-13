import os
import sys
import threading
import time
import cv2
import numpy as np

from jetson_server.conf.config import hostname, port
from jetson_server.SocketServer import SocketServer
from jetson_server.conf.config import model_path, min_thresh
from jetson_server.YoloDetector import YoloDetector
from jetson_server.LatestFrameBuffer import LatestFrameBuffer, FramePacket
from shared.logs.logs import Logger
from shared.protocol.detection_schema import DetectionMessage

def launch_reception_thread(server, latest_frame_buffer):
    # 1. Tant que le serveur tourne et que le client est connecté :
    while server.is_running():
        # a. recevoir une frame
        payload = server.receive_message()

        if payload is None:
            logger.info("Connexion fermée par le client.")
            break

        # b. transformer payload -> image numpy
        nparr = np.frombuffer(payload, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # d. latest_frame_buffer.put(frame_packet)
        latest_frame_buffer.update_frame(FramePacket(frame_data=frame))

        logger.info(f"Message reçu : Taille du payload {len(payload)} octets et frame de dimensions {frame.shape if frame is not None else 'N/A'}")

def launch_inference_thread(server, latest_frame_buffer, yolo_detector):
    # 1. Tant que le serveur tourne et que le client est connecté :
    while server.is_running():
    # a. attendre une frame disponible
        frame_packet = latest_frame_buffer.get_latest_frame()
        if frame_packet is None:
            continue
        # b. faire l’inférence pour obtenir les détections
        detections = yolo_detector.detect_objects(frame_packet.frame_data)
        # c. construire DetectionMessage
        detection_message = DetectionMessage(frame_id=frame_packet.frame_id, detections=detections)
        # d. envoyer le message au client
        server.send_detection_message(detection_message)

if __name__ == "__main__":
    logger = Logger();
    server = None
    try:
        # 1. Créer LatestFrameBuffer
        latest_frame_buffer = LatestFrameBuffer()

        # 2. Charger YoloDetector et faire le warmup
        yolo_detector = YoloDetector(model_path, min_thresh)

        # 3. Démarrer le serveur réseau
        server = SocketServer(host=hostname, port=port)

        server.start()
        server.accept_client()

        # 5. Lancer le thread de réception
        reception_thread = threading.Thread(target=launch_reception_thread, args=(server, latest_frame_buffer))
        # 6. Lancer le thread d’inférence
        inference_thread = threading.Thread(target=launch_inference_thread, args=(server, latest_frame_buffer, yolo_detector))

        inference_thread.start()
        reception_thread.start()

        inference_thread.join()
        reception_thread.join()

    except KeyboardInterrupt:
        logger.info("Arrêt du serveur demandé par l'utilisateur.")
    except Exception as e:
        logger.error(f"Erreur serveur : {e}")
        sys.exit(1)
    finally:
        if server is not None:
            server.stop()