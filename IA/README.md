labelisation : https://github.com/developer0hye/Yolo_Label



Exporter modèle sur rasberi pi 5 imx500

Installer les outils côté Raspberry Pi 5


sudo apt update && sudo apt full-upgrade
sudo apt install imx500-all    # firmware + packager + post-traitements
sudo reboot


Obtenir packerOut.zip à partir de ton model.onnx
executer export.py


Packager sur le Raspberry Pi → network.rpk
Transfère packerOut.zip sur le Pi puis:



imx500-package -i /chemin/vers/packerOut.zip -o out_dir
# génère out_dir/network.rpk
Doit être exécuté sur le Pi. ([Raspberry Pi][5])
Docs packager officielles si besoin. ([developer.aitrios.sony-semicon.com][6])

Charger et exécuter sur l’IMX500



Picamera2 (Python)

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500

picam = Picamera2()
imx = IMX500("/chemin/out_dir/network.rpk")
imx.show_network_fw_progress_bar()   # suivi du chargement firmware
picam.start()
# récupère les tensors via picam.metadata, puis post-traiter

