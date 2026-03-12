import os
import sys
import threading
import time

from jetson_server.SocketServer import SocketServer
from jetson_server.conf.config import model_path, min_thresh
from jetson_server.YoloDetector import YoloDetector
from jetson_server.LatestFrameBuffer import LatestFrameBuffer
from shared.logs.logs import Logger

def launch_reception_thread(con, latest_frame_buffer):
    # 1. Tant que le serveur tourne et que le client est connecté :
    while con.is_running():
    # a. recevoir une frame
        frame = con.receive_frame()
    # b. créer un FramePacket(frame_id, timestamp, frame)
        frame_packet = FramePacket(frame_id, timestamp, frame)
    # c. déposer/remplacer dans LatestFrameBuffer
        latest_frame_buffer.update_frame(frame_packet)

def launch_inference_thread(con, latest_frame_buffer, yolo_detector):
    # 1. Tant que le serveur tourne et que le client est connecté :
    while con.is_running():
    # a. attendre une frame disponible
        frame_packet = latest_frame_buffer.get_latest_frame(block=True, timeout=1)
        if frame_packet is None:
            continue
    # b. récupérer le dernier FramePacket
    # c. exécuter l’inférence
        detections = yolo_detector.detect_objects(frame_packet.frame_data)
    # d. construire DetectionMessage
        detection_message = yolo_detector.build_detection_message(detections, frame_packet)
    # e. envoyer le message au client
        con.send_detection_message(detection_message)

def stop_inference_server(con, reception_thread, inference_thread):
    pass

if __name__ == "__main__":
    logger = Logger();

    try:
        # 1. Créer LatestFrameBuffer
        latest_frame_buffer = LatestFrameBuffer()

        # 2. Charger YoloDetector et faire le warmup
        yolo_detector = YoloDetector(model_path, min_thresh)

        # 3. Démarrer le serveur réseau
        con = SocketServer()
        logger.info("Attente d'une connection cliente ...")
        con.start()
        
        if not con.is_running():
            logger.error("Le serveur n'a pas pu démarrer !, fermeture du serveur ...")
            sys.exit(1)
        con.show_status()
        i = 0
        while True and i < 10:
            con.receive_frame()
            time.sleep(1)
            i += 1

        """
        # 5. Lancer le thread de réception
        reception_thread = threading.Thread(target=launch_reception_thread, args=(con, latest_frame_buffer))
        reception_thread.start()
        
        # 6. Lancer le thread d’inférence
        inference_thread = threading.Thread(target=launch_inference_thread, args=(con, latest_frame_buffer, yolo_detector))
        inference_thread.start()

        reception_thread.join()
        inference_thread.join()
        """
        con.end() 
        if con.is_running():
            logger.error(f"Le serveur n'a pas pu se fermer correctement !")
            sys.exit(1)
        con.show_status()
    except Exception as e:
        logger.error(e)
        sys.exit(1)