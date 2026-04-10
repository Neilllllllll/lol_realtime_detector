import cv2
import os
import json

def decouper(video,dossier_sortie,intervalle_fps,coordonnnees_bbox):


    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)
        print(f"Dossier '{dossier_sortie}' créé.")

    video = cv2.VideoCapture(video)

    if not(video.isOpened()):
        print("Erreur video.")


    #FPS de la video
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📊 Infos vidéo : {fps} FPS | {total_frames} frames au total.")
    print(f"✂️ Découpage en cours... (1 image toutes les {intervalle_fps} frames)")

    frame_count = 0
    images_sauvegardees = 0


    while True:
        ret, frame = video.read()

        if not ret:
            break

        if frame_count % intervalle_fps == 0:
            nom_image = os.path.join(dossier_sortie, f"frame_{images_sauvegardees:04d}.jpg")
            cv2.imwrite(nom_image, frame)
            img_path = os.path.join(dossier_sortie, f"frame_{images_sauvegardees:04d}.jpg")
            txt_name = os.path.splitext(f"frame_{images_sauvegardees:04d}.jpg")[0] + '.txt'
            txt_path = os.path.join(dossier_sortie, txt_name)
            with open(txt_path, 'w') as f:
                f.write(f"{coordonnnees_bbox}\n") #Coordonnées de la bounding box (x_center, y_center, width, height) manuel;
            images_sauvegardees += 1
        frame_count += 1


    video.release()
    print("video terminer")


if __name__ == "__main__":
    with open("./champions.json") as f:
        data = json.load(f)

    for champion in data["champions"]:
        video_path = "videos/" + champion + ".mp4"
        output_folder = f"frames_{champion}"
        coordonnnees_bbox = data["champions"][champion]
        print(f"Traitement de la vidéo de {champion}...")
        print(f"Coordonnées de la bounding box pour {champion} : {coordonnnees_bbox}")
        decouper(video_path, output_folder, 30, coordonnnees_bbox)#Lancement decoupage de la vidéo avec un intervalle de 30 frames
        


