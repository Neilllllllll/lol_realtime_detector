import os
import time
import cv2
import json
from shared.logs.logs import Logger
from pc_client.SocketClient import SocketClient

def main():
    logger = Logger()
    host, port = ()
    client = SocketClient(host, port)

    images_folder = "pc_client/images_test"
    tentatives_connexion = 5

    while tentatives_connexion > 0:
        try:
            client.connect()
            logger.info("Client connecté au serveur.")
            break
        except Exception as e:
            logger.error(f"Échec de la connexion : {e}")
            tentatives_connexion -= 1
            logger.info(f"Nouvelle tentative dans 5 secondes... ({tentatives_connexion} tentatives restantes)")
            time.sleep(5)

    try:
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

            response_data = json.loads(response.decode("utf-8"))

            if response_data["detections"]:
                cv2.rectangle(
                    frame, 
                    (response_data["detections"][0]["bbox"][0], response_data["detections"][0]["bbox"][1]), 
                    (response_data["detections"][0]["bbox"][0] + response_data["detections"][0]["bbox"][2], response_data["detections"][0]["bbox"][1] + response_data["detections"][0]["bbox"][3]), 
                    (0, 255, 0), 2
                )
            cv2.imshow("Image", frame)
            cv2.waitKey(1000)   
            cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Erreur côté client : {e}")
    finally:
        client.close()
        logger.info("Client fermé proprement.")

if __name__ == "__main__":
    main()