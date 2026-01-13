## StereoAngleCalculator (stéréo → 3D → angles)

### Objectif

Calculer la position 3D d’un point à partir de deux caméras (gauche/droite), puis en déduire :

* Pitch (vertical)
* Yaw (horizontal)
* Distance réelle

Les calculs sont basés sur le schéma : 
![Schéma de profondeur](img/shemaProfondeur.png)

## Classe `StereoAngleCalculator`

```python
StereoAngleCalculator(
    baseline_m=0.10,
    h_cl_m=0.0,
    focal_length_mm=4.74,
    pixel_size_mm=0.00155
)
```

### Paramètres

* `baseline_m` : distance entre les deux caméras (m)
* `h_cl_m` : décalage vertical caméra → laser (m)
* `focal_length_mm` : focale
* `pixel_size_mm` : taille d’un pixel

## Fonctions

### `load_stereo_calibration(path)`

Charge une calibration stéréo `.npz`.

Met à jour :

* Maps de rectification
* Intrinsèques `fx, fy, cx, cy`
* Baseline réelle (via T)

### `set_image_size(width, height)`

Fallback si pas de calibration.
Calcule automatiquement :

* `fx, fy` à partir des specs caméra
* `cx, cy` au centre de l’image

### `compute_angles(lx, ly, rx, ry)`

À partir des coordonnées pixel gauche/droite, calcule :

* Position 3D (X, Y, Z)
* Pitch (°)
* Yaw (°)
* Distance (m)

Retour :

```python
pitch_deg, yaw_deg, distance_m
```

Principe :

1. Disparité = lx - rx (si négatve, pas point correcte car object à gauche sera toujours plus grand)
2. Triangulation stéréo
3. Projection 3D
4. Conversion en angles (voir `img/shemaProfondeur.png`)

## Sorties

* `pitch` : angle vertical pour le galvo
* `yaw` : angle horizontal pour le galvo
* `distance` : distance réelle au point

