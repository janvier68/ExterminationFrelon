import cv2

def extraire_images(video_path, output_folder, nb_images):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duree = total_frames / fps

    # Intervalle en secondes entre chaque image à extraire
    # intervalle = duree / 277 #(4*60+37)
    # intervalle = int(round(fps,0))
    # print(intervalle)

    for i in range(nb_images):
        temps = i #* intervalle
        frame_id = int(temps * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/image_{i+1:03d}.jpg", frame)

    cap.release()

# Exemple d'utilisation
extraire_images("GOPR0103.MP4", "output", nb_images=200)
