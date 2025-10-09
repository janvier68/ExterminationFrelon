from ultralytics import YOLO

def detect_video(input_path, output_path, model_path, device=0):
    # Charger le modèle entraîné
    model = YOLO(model_path)
    
    # Lancer l’inférence sur la vidéo
    results = model.predict(
        source=input_path,   # vidéo en entrée
        save=True,           # sauvegarde la vidéo annotée
        project="runs/detect",  # dossier de sortie
        name="predict_custom",  # sous-dossier
        exist_ok=True,       # écraser si existe déjà
        device=device        # 0 = GPU, "cpu" = CPU
    )
    
    # La vidéo sera sauvegardée automatiquement dans runs/detect/predict_custom/
    print("Vidéo détectée enregistrée dans :", results[0].save_dir)

if __name__ == "__main__":
    input_video = "IMG_6876.mp4"
    output_video = "output_detected.mp4"  # le nom réel sera généré par Ultralytics
    model_path = "best.pt"
    
    detect_video(input_video, output_video, model_path, device=0)
