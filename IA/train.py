from ultralytics import YOLO

# Charger le modèle pré-entraîné YOLOv11s
model = YOLO("yolo11n.pt")

# Entraîner le modèle
results = model.train(
    data="dataset/data.yaml",
    epochs=100,          # à ajuster selon ton matériel
    imgsz=640,          # taille des images d'entrée
    batch=16,           # taille du batch
    name="bee_hornet_yolo11nV4",
    project="runs/train"
)
