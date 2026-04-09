import cv2
import os
import json

def decouper(video,dossier_sortie,intervalle_fps):


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
            images_sauvegardees += 1
        frame_count += 1


    video.release()
    print("video terminer")


if __name__ == "__main__":
    with open("../champion.json") as f:
        data = json.load(f)

    for champion in data["champions"]:
        video_path = "videos/" + champion + ".mp4"
        output_folder = f"frames_{champion}"
        decouper(video_path, output_folder, 30)



