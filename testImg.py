from ultralytics import YOLO

# Charger ton modèle entraîné
model = YOLO("best.pt")

# Inférence sur une image
results = model("image.jpeg", save=True, project="runs/inference", name="test1")

# 'save=True' → sauvegarde une copie annotée dans runs/detect/predict/
# 'show=True' → ouvre une fenêtre avec l'image annotée
