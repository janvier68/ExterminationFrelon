# from ultralytics import YOLO

# model = YOLO("best.pt")

# # device=0 force le GPU. int8 et nms requis par IMX.
# model.export(
#     format="imx",
#     data="dataset/data.yaml",
#     device=0,          # GPU
#     int8=True,         # requis par IMX
#     nms=True,          # requis par IMX
#     imgsz=640,         # ajuste si besoin
# )

# # Load the exported model
# imx_model = YOLO("yolo11n_imx_model")

# # Run inference
# results = imx_model("https://ultralytics.com/images/bus.jpg")



from ultralytics import YOLO
m = YOLO("best.pt")          # ou un .pt YOLO
m.export(format="imx")       # produit un dossier avec packerOut.zip
