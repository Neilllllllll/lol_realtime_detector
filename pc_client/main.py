import os
import time
import cv2

from shared.logs.logs import Logger
from pc_client.SocketClient import SocketClient

def main():
    logger = Logger()
    host, port = ("127.0.0.1", 5596)
    client = SocketClient(host, port)

    images_folder = "pc_client/images_test"

    try:
        client.connect()
        logger.info("Client connecté au serveur.")

        for image in os.listdir(images_folder):
            image_path = os.path.join(images_folder, image)

            frame = cv2.imread(image_path)
            if frame is None:
                logger.error(f"Impossible de lire l'image : {image_path}")
                continue

            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                logger.error(f"Impossible d'encoder l'image : {image_path}")
                continue

            payload = buffer.tobytes()
            logger.info(f"Envoi de {image} ({len(payload)} octets)")
            client.send_message(payload)

            response = client.receive_message()
            if response is None:
                logger.info("Connexion fermée par le serveur.")
                break

            logger.info(f"Réponse reçue : {response.decode('utf-8')}")

            time.sleep(1)

    except Exception as e:
        logger.error(f"Erreur côté client : {e}")
    finally:
        client.close()
        logger.info("Client fermé proprement.")

if __name__ == "__main__":
    main()