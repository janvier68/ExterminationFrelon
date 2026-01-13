# Partie IA

## 1. Présentation

Cette partie du projet concerne la préparation des données, l'entraînement d'un modèle YOLO, l'augmentation du dataset, puis l'export et le déploiement du modèle sur un **Raspberry Pi 5 avec IMX500**.

Le dépôt de labellisation utilisé :
[dataset roboflow](https://universe.roboflow.com/toan-ydjjh/bee-management-system-2angl/dataset/2)
on à rajouter notre propre labélisation avec https://github.com/developer0hye/Yolo_Label


## 2. Structure des outils

### 2.1 augmente.py

Script permettant d'augmenter un dataset par :

* Zoom
* Tuilage (tiling)
* Génération d'images synthétiques

#### Utilisation

```bash
python augmente.py --in_dir origineLabel --out_dir dataAugmented --count 3000
```

#### Paramètres

* `--in_dir` : dossier contenant les images et labels d'origine
* `--out_dir` : dossier de sortie pour les données augmentées
* `--count` : nombre total d'images générées

### 2.2 export.py

Script permettant de convertir un modèle YOLO (.pt ou .onnx) vers un format compatible IMX500.
Il génère un fichier `packerOut.zip` utilisable par les outils Sony IMX500.

### 2.3 newDataset.sh

Script shell permettant de :

* Dézipper un nouveau dataset
* L'intégrer dans un dataset existant
* Respecter la structure :

```
train/
 ├── images/
 └── labels/
val/
 ├── images/
 └── labels/
test/
 ├── images/
 └── labels/
```

### 2.4 splitVideoImage

Script permettant d'extraire des images depuis une vidéo à un intervalle donné.

Exemple : 1 image toutes les X secondes.

### 2.5 train.py

Script d'entraînement du modèle YOLO à partir du dataset préparé.

Fonctionnalités :

* Chargement du dataset
* Entraînement
* Sauvegarde des poids
* Export possible vers ONNX

## 3. Pipeline IA complet

### Étape 1 — Préparation des données

1. Labellisation via Yolo_Label
2. Augmentation du dataset (augmente.py)
3. Fusion de datasets si nécessaire (newDataset.sh)

### Étape 2 — Entraînement

Lancer l'entraînement :

```bash
python train.py
```

À la fin, on obtient un modèle :`.pt`

## 4. Export du modèle pour IMX500 (Raspberry Pi 5)

### 4.1 Génération du package IMX500

À partir de ton modèle `.pt` :

```bash
python export.py
```

Cela génère best_imx_model avec:

```
packerOut.zip
```

### 4.2 Transfert vers le Raspberry Pi

Copier sur le Raspberry Pi :

* `packerOut.zip`
* `label.txt`

Dans le dossier :

```
Raspberry/IA/
```

## 5. Installation côté Raspberry Pi 5

### 5.1 Mise à jour système

```bash
sudo apt update && sudo apt full-upgrade
```

### 5.2 Installation des outils IMX500

```bash
sudo apt install imx500-all
sudo reboot
```

Cela installe :

* Firmware IMX500
* Outils de packaging
* Post-traitements

## 6. Déploiement du modèle

une fois bien mis, vous pourrez le voire en action dans main.py

## 7. Dépendances

* Python 3.8+
* YOLO (Ultralytics ou autre)
* OpenCV
* Numpy
* Outils Sony IMX500
