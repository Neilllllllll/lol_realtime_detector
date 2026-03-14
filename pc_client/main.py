import cv2
import numpy as np
import pyautogui
import json
import time
from PIL import ImageGrab
from pc_client.SocketClient import SocketClient
from pc_client.conf.config import hostname, port
from shared.logs.logs import Logger


def get_screenshot() -> np.ndarray:
    width, height = 1920, 1080
    screenshot = ImageGrab.grab(bbox=(0, 0, width, height))
    screenshot_converted = np.array(screenshot)
    screenshot_converted = cv2.cvtColor(screenshot_converted, cv2.COLOR_RGB2BGR)
    return screenshot_converted

def connect_to_server(host: str, port: int) -> SocketClient:
    logger = Logger()
    client = SocketClient(host, port)
    tentatives_connexion = 5

    while tentatives_connexion > 0:
        try:
            client.connect()
            logger.info("Client connecté au serveur.")
            return client
        except Exception as e:
            logger.error(f"Échec de la connexion : {e}")
            tentatives_connexion -= 1
            logger.info(f"Nouvelle tentative dans 5 secondes... ({tentatives_connexion} tentatives restantes)")
            time.sleep(5)

    raise ConnectionError("Impossible de se connecter au serveur après plusieurs tentatives.")

def main(): 
    loop_time = time.time()
    logger = Logger()
    fps = loop_time
    i = 0
    client = connect_to_server(hostname, port)
    while(True):
        loop_time = time.time()
        logger.info("Début de la capture écran...")
        screenshot_converted = get_screenshot()

        payload = cv2.imencode(".jpg", screenshot_converted, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1].tobytes()
        client.send_message(payload)

        response = client.receive_message()
        if response is None:
            logger.info("Connexion fermée par le serveur.")
            break

        response_data = json.loads(response.decode("utf-8"))

        if i > 10:
            fps = 1 / (time.time() - loop_time)
            i = 0
        else:
            i += 1

        if response_data["detections"]:
            cv2.rectangle(
                screenshot_converted, 
                (response_data["detections"][0]["bbox"][0], response_data["detections"][0]["bbox"][1]), 
                (response_data["detections"][0]["bbox"][0] + response_data["detections"][0]["bbox"][2], response_data["detections"][0]["bbox"][1] + response_data["detections"][0]["bbox"][3]), 
                (0, 255, 0), 2
            )

        cv2.putText(screenshot_converted, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Screenshot", screenshot_converted)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

