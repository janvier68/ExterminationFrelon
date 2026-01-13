# 🐝 Projet Cap 5A — Protection de ruches contre les frelons par laser automatisé

## Objectif

Les frelons asiatiques se positionnent à l’entrée des ruches, empêchant les abeilles de sortir et entraînant leur épuisement et leur mort.
Ce projet vise à **détecter automatiquement les frelons** à l’aide d’une IA embarquée et à les **repousser avec un laser piloté par galvos**, afin de protéger les ruches.

## Architecture globale
 - IA
    entrainrer le modèle pour reconnaitre le frelon
 - Rasbery
    code récuprer infot caméra calcule angle galvo et alume le laser

## Liste du matériel

* Raspberry Pi 5 — 8 Go
* 2 x Raspberry Pi AI Camera (IMX500)
* Module galvo laser
  Produit : [https://www.amazon.fr/dp/B0C7VLWXV3](https://www.amazon.fr/dp/B0C7VLWXV3)
  Alimentation
*  2x ....
*  1x
* Laser (attention c'est dangereux)
* Boîtier


## Installation du projet

### Entraînement de l’IA
nous avons déjà entrainter un modèle yolo11n avec notre propre dataset, il se trouce ici IA/best.py

#### Étapes

1. Créer un dataset (frelon / abeille / fond)
2. Fine-tuning du modèle
3. Export compatible IMX500
4. Conversion au format requis par la NPU
5. Déploiement sur la Raspberry

detailler dans [[IA/README.md]]



## 1. Installation du système d’exploitation

il faut maitre le systemte d'exploitation de la rasbery sur un carte sd (min 16go) pour cela nous allons utiliser la facher officher rasbery

### 1.1 Télécharger Raspberry Pi Imager

Télécharger l’outil officiel ici :
[https://www.raspberrypi.com/software/](https://www.raspberrypi.com/software/)

Installer Raspberry Pi Imager sur votre ordinateur (Windows / Mac / Linux).

### 1.2 Choix du système

1. Ouvrir **Raspberry Pi Imager**
2. Cliquer sur **Choose OS**
3. Aller dans :
   * Raspberry Pi OS (other)
   * Puis sélectionner :
     **Raspberry Pi OS (Legacy) Lite 64bit**

⚠️ Ce choix est important car cette version contient Python 3.11 natif, nécessaire pour le projet.

### 1.3 Choix de la carte SD

1. Cliquer sur **Choose Storage**
2. Sélectionner votre carte SD

### 1.4 Paramètres avancés (IMPORTANT)

Avant de flasher, cliquer sur l’icône ⚙️ (roue dentée) et configurer :
✅ Activer SSH
✅ Définir un nom d’utilisateur et mot de passe
✅ Configurer le Wi-Fi
✅ Régler le fuseau horaire

Cela permet de se connecter à la Raspberry sans écran.

### 1.5 Flash de la carte SD

Cliquer sur **Write** et attendre la fin de l’écriture.

## 2. Démarrage de la Raspberry Pi

1. Insérer la carte SD dans la Raspberry Pi
2. Brancher l’alimentation
3. Attendre environ 1 minute

## 3. Connexion en SSH

Depuis votre ordinateur :

### Windows

Utiliser PowerShell ou Putty.

### Mac / Linux

Ouvrir un terminal.

Commande :

```bash
ssh utilisateur@ip_de_la_raspberry
```

Exemple :

```bash
ssh pi@192.168.1.42
```

## 4. Mise à jour du système

Une fois connecté :

```bash
sudo apt update
sudo apt install git
```

Cela peut prendre plusieurs minutes.

## 5. Installation du projet

### 5.1 Télécharger le projet

```bash
git clone https://github.com/janvier68/ExterminationFrelon.git
```

Puis :

```bash
cd ExterminationFrelon/Rasbery
```

### 5.2 Création d’un environnement Python

Cela évite de casser le système.

```bash
python3 -m venv venv
```

Activation :

```bash
source venv/bin/activate
```

Après activation, vous verrez `(venv)` devant la ligne de commande.

### 5.3 Activation automatique du venv (optionnel mais recommandé)

Éditer le fichier `.bashrc` :

```bash
nano ~/.bashrc
```

Ajouter à la fin :

```bash
source ~/ExterminationFrelon/Rasbery/venv/bin/activate
```

Sauvegarder :
CTRL + O → Entrée
Quitter :
CTRL + X

## 6. Installation des dépendances

Toujours dans le dossier `Rasbery` et avec le venv activé :

```bash
pip install -r requirements.txt
```

Cela installe :
* OpenCV
* Flask
* Librairies IMX500
* Outils mathématiques
* Etc.

## 7. Installation du modèle IA

1. Récupérer le modèle YOLO11n compatible IMX500
2. Copier le fichier dans le dossier prévu (ex : `models/`)
3. Vérifier qu’il est bien reconnu par la NPU

## 8. Setup matériel

Lancer :

```bash
python setup.py
```

Ce script va :
* Vous demander les informations sur la ruche
* Tester la caméra
* Tester le laser
* Tester les galvos
* Vérifier les connexions
* Lancer la calibration

### Calibration caméra

Vous aurez besoin d’un damier imprimé :
`docs/Dammier.png`

Imprimer ce fichier et le placer devant la caméra pendant la calibration.

## 9. Lancement du système

### Version avec interface web

```bash
python main.py
```

Accès via navigateur :

```
http://ip_de_la_raspberry:5000
```

### Version sans interface (mode autonome)

```bash
python mainNoUI.py
```

## Explication technique

### Calcul de profondeur

La profondeur est calculée par trigonométrie.

Schéma :
`docs/schemaProfondeur.png`

### Caméra

Documentation officielle :
[https://www.raspberrypi.com/documentation/accessories/ai-camera.html](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)

Utilisation :
* OpenCV pour calibration
* Calibration mono et stéréo
* Exploitation de la NPU IMX500
* YOLO11n optimisé pour performance temps réel

### Galvos

Code basé sur :
[https://www.instructables.com/Arduino-Laser-Show-With-Real-Galvos/](https://www.instructables.com/Arduino-Laser-Show-With-Real-Galvos/)

Rôle :

* Convertir les coordonnées IA → angles
* Piloter précisément le laser
* Correction géométrique


## Sécurité

⚠️ Le laser doit être :

* Classe faible puissance
* Jamais dirigé vers les humains
* Jamais vers les abeilles
* Limité à une zone définie

Une **safe zone** en pixels est définie.


## TODO

### Logiciel

* [ ] Configuration safe zone en pixels
* [ ] `pip freeze` → liste exacte des dépendances
* [ ] Refactorisation du code
* [ ] Tests :
  * setup.py
  * mainNoUI.py
  * testMateriel.py
* [ ] Amélioration de l’algorithme

### Documentation

* [ ] Datasheet caméra
* [ ] Datasheet galvo
* [ ] Guide IA complet
* [ ] Guide installation Raspberry
* [ ] Explication architecture
* [ ] Schéma du boîtier
* [ ] Organisation interne
