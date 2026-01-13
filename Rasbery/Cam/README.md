## CameraManager & YOLO (IMX500)

### Objectif

Interface simple pour utiliser un modèle YOLO packagé IMX500 avec une caméra Sony (`AiCamera`) :

* Détection en temps réel
* Annotation des bounding boxes
* Flux MJPEG
* Filtrage via une safe zone centrale


## Classe `YOLO`

Wrapper autour de `modlib.models.Model`.

Fonctions :

* Charge `packerOut.zip`
* Charge `label.txt`
* Applique le post-process YOLO Ultralytics
* Fournit le centre d’une bounding box

```python
YOLO(pathZip, pathLab)
```


## Classe `CameraManager`

Gère la caméra, le modèle, l’annotation et le filtrage.

```python
CameraManager(
    pathZip,
    pathLab,
    cam_index=0,
    frame_rate=8,
    safeZone=0.20,
    on_detection=None
)
```

### Attributs principaux

* `safeZone` : pourcentage de padding sur chaque bord
* `on_detection` : callback recevant les centres détectés
* `device` : instance `AiCamera`
* `model` : instance `YOLO`


## Fonctions

### `generate_frames()`

Générateur MJPEG.

* Lit les frames caméra
* Annote les bounding boxes
* Dessine la safe zone
* Appelle `on_detection(centers)` si défini
* Retourne un flux MJPEG

Utilisation typique : streaming Flask / FastAPI.


### `get_detections(score_threshold=0.5)`

Retourne une liste de détections filtrées.

Filtrage :

* Score ≥ threshold
* Centre de la bbox dans la safe zone

Format de sortie :

```python
[
  {
    "score": float,
    "class_id": int,
    "label": str,
    "center": (x, y)
  }
]
```


### `stop()`

Arrête la caméra.


## Safe Zone

Rectangle central utilisé pour ignorer les détections trop proches des bords.
Calculé comme un padding proportionnel à la taille de l’image.
