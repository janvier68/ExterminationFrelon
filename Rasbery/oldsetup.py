
#!/usr/bin/env python3
# orchestrator_calibration.py
#
# Flow:
# 1) Demande à l’utilisateur: baseline (m), nb photos, délai (s), pattern damier, taille carré (m)
# 2) Lance l’UI de capture (basé sur photoForCalib.py) en lui passant N_PHOTOS + COUNTDOWN_SECONDS
# 3) Lance la calibration stéréo (basé sur calibrationCamera.py) -> produit stereo_params.npz
# 4) Vérifie la qualité (RMS <= seuil). Si NOK: recommence capture + calibration
# 5) Alignment laser: affiche une mire (carré max + croix centre), stream cam gauche avec centre.
#    L’utilisateur ajuste physiquement le laser pour que centre laser = centre caméra.
# 6) Demande: distance mur (m) + largeur/hauteur rectangle laser (m)
#    Calcule max_angle_deg et écrit dans le JSON + paths.

import os
import sys
import json
import time
import math
import shutil
import signal
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2

# ====== chemins vers TES scripts (fichiers fournis) ======
CAPTURE_UI_PY = "Config/photoForCalib.py"         # :contentReference[oaicite:0]{index=0}
STEREO_CALIB_PY = "Config/calibrationCamera.py"  # :contentReference[oaicite:1]{index=1}

# ====== config ======
CONFIG_JSON_PATH = "Config/config.json"
DEFAULT_CALIB_ROOT = "Config/photos_calibration"  # photoForCalib.py crée photos_calibration, photos_calibration1, etc. :contentReference[oaicite:2]{index=2}

RMS_OK_THRESHOLD = 1.5  # à ajuster selon ton setup (plus bas = plus strict)

# ====== laser mire (aide visuelle) ======
MIRE_WIN = "Mire Laser (aide alignment)"
CAM_WIN = "Cam Gauche (centre)"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def die(msg: str, code: int = 1):
    raise SystemExit(f"[ERREUR] {msg}")


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        die(f"Impossible de lire {path}: {e}")


def save_json(path: str, data: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def ensure_dict(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in d or not isinstance(d[key], dict):
        d[key] = {}
    return d[key]


def input_float(prompt: str, default: Optional[float] = None, min_v: Optional[float] = None) -> float:
    while True:
        s = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + " : ").strip()
        if not s and default is not None:
            v = float(default)
        else:
            try:
                v = float(s)
            except ValueError:
                print("Nombre invalide.")
                continue
        if min_v is not None and v < min_v:
            print(f"Doit être >= {min_v}")
            continue
        return v


def input_int(prompt: str, default: Optional[int] = None, min_v: Optional[int] = None) -> int:
    while True:
        s = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + " : ").strip()
        if not s and default is not None:
            v = int(default)
        else:
            try:
                v = int(s)
            except ValueError:
                print("Entier invalide.")
                continue
        if min_v is not None and v < min_v:
            print(f"Doit être >= {min_v}")
            continue
        return v


def run_python(script_path: str, env: Optional[Dict[str, str]] = None) -> int:
    if not os.path.exists(script_path):
        die(f"Script introuvable: {script_path}")
    cmd = [sys.executable, script_path]
    p = subprocess.Popen(cmd, env=env)
    return p.wait()


def find_latest_calib_dir(root_prefix: str = DEFAULT_CALIB_ROOT) -> str:
    """
    photoForCalib.py crée photos_calibration, puis photos_calibration1, photos_calibration2, ...
    On prend le plus récent existant.
    """
    candidates = []
    if os.path.exists(root_prefix):
        candidates.append(root_prefix)
    i = 1
    while True:
        d = f"{root_prefix}{i}"
        if not os.path.exists(d):
            break
        candidates.append(d)
        i += 1
    if not candidates:
        die("Aucun dossier de capture trouvé (photos_calibration*).")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def parse_rms_from_stdout(stdout: str) -> Optional[float]:
    # calibrationCamera.py imprime: "[INFO] Calibration stéréo OK, RMS = <val>" :contentReference[oaicite:3]{index=3}
    key = "RMS ="
    for line in stdout.splitlines():
        if key in line:
            try:
                return float(line.split(key, 1)[1].strip())
            except Exception:
                return None
    return None


def run_stereo_calib(calib_dir: str, pattern_w: int, pattern_h: int, square_size_m: float) -> Dict[str, Any]:
    """
    Lance calibrationCamera.py mais en le COPIANT puis patchant 3 constantes simples:
    - CALIBFILE
    - PATTERN_SIZE
    - SQUARE_SIZE
    Pour éviter de modifier ton fichier source à la main.
    """
    work = os.path.abspath(".work_calib_tmp")
    os.makedirs(work, exist_ok=True)
    patched = os.path.join(work, "stereo_calib_patched.py")
    shutil.copy(STEREO_CALIB_PY, patched)  # :contentReference[oaicite:4]{index=4}

    with open(patched, "r", encoding="utf-8") as f:
        src = f.read()

    src = src.replace('CALIBFILE = "photos_calibration"', f'CALIBFILE = "{calib_dir}"')
    src = src.replace("PATTERN_SIZE = (9, 6)", f"PATTERN_SIZE = ({pattern_w}, {pattern_h})")
    # calibrationCamera.py utilise SQUARE_SIZE en "m" si tu mets m ; ici on impose mètres.
    src = src.replace("SQUARE_SIZE  = 0.025", f"SQUARE_SIZE  = {square_size_m}")

    with open(patched, "w", encoding="utf-8") as f:
        f.write(src)

    cmd = [sys.executable, patched]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate()

    rms = parse_rms_from_stdout(out)
    if p.returncode != 0:
        die(f"Calibration stéréo échouée.\n--- log ---\n{out}\n--- fin log ---")

    # calibrationCamera.py déplace stereo_params.npz vers <calib_dir>/visualisation_calibration/stereo_params.npz :contentReference[oaicite:5]{index=5}
    stereo_npz = os.path.join(calib_dir, "visualisation_calibration", "stereo_params.npz")
    if not os.path.exists(stereo_npz):
        die(f"stereo_params.npz introuvable: {stereo_npz}")

    return {"rms": rms, "stereo_npz": stereo_npz, "log": out}


def launch_capture_ui(n_photos: int, delay_s: float) -> None:
    """
    photoForCalib.py a N_PHOTOS et COUNTDOWN_SECONDS en dur :contentReference[oaicite:6]{index=6}
    => on fait un patch temporaire identique au principe calibration.
    """
    work = os.path.abspath(".work_capture_tmp")
    os.makedirs(work, exist_ok=True)
    patched = os.path.join(work, "capture_ui_patched.py")
    shutil.copy(CAPTURE_UI_PY, patched)  # :contentReference[oaicite:7]{index=7}

    with open(patched, "r", encoding="utf-8") as f:
        src = f.read()

    # Remplacements simples sur les lignes "N_PHOTOS = 120" et "COUNTDOWN_SECONDS = 0.2" :contentReference[oaicite:8]{index=8}
    src = src.replace("N_PHOTOS = 120", f"N_PHOTOS = {int(n_photos)}")
    src = src.replace("COUNTDOWN_SECONDS = 0.2", f"COUNTDOWN_SECONDS = {float(delay_s)}")

    with open(patched, "w", encoding="utf-8") as f:
        f.write(src)

    print("\n1) Capture: ouvrir http://0.0.0.0:5000 dans un navigateur")
    print("2) Lancer la séquence, attendre fin, puis CTRL+C dans le terminal pour revenir ici.\n")

    # Process bloquant: l'utilisateur stoppe au CTRL+C
    subprocess.run([sys.executable, patched], check=False)


def draw_mire(w: int, h: int) -> Any:
    img = 255 * (0 * (cv2.UMat(h, w, cv2.CV_8UC3)))  # placeholder
    img = cv2.UMat.get(img)
    img[:] = (0, 0, 0)

    cx, cy = w // 2, h // 2
    # croix centre
    cv2.line(img, (cx - 60, cy), (cx + 60, cy), (255, 255, 255), 2)
    cv2.line(img, (cx, cy - 60), (cx, cy + 60), (255, 255, 255), 2)

    # carré "max" (juste visuel)
    margin = int(min(w, h) * 0.15)
    cv2.rectangle(img, (margin, margin), (w - margin, h - margin), (255, 255, 255), 2)

    cv2.putText(img, "Ajuste le laser: centre laser == centre camera gauche",
                (20, 40), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Appuie sur Q pour quitter la mire",
                (20, 75), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img

# ? faut le fair sur le flask
def laser_alignment_loop(cam_index_left: int = 0):
    cap = cv2.VideoCapture(cam_index_left)
    if not cap.isOpened():
        die(f"Impossible d'ouvrir la caméra index={cam_index_left}")

    # tente de récupérer une taille d'image
    ret, frame = cap.read()
    if not ret:
        cap.release()
        die("Impossible de lire un frame caméra.")
    h, w = frame.shape[:2]

    mire = draw_mire(w, h)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # centre caméra gauche
        cx, cy = w // 2, h // 2
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
        cv2.putText(frame, f"Centre cam: ({cx},{cy})", (10, h - 10), FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(MIRE_WIN, mire)
        cv2.imshow(CAM_WIN, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyWindow(MIRE_WIN)
    cv2.destroyWindow(CAM_WIN)


def compute_max_angle_deg(distance_m: float, rect_w_m: float, rect_h_m: float) -> float:
    """
    Hypothèse: max_angle est le max entre demi-largeur et demi-hauteur:
      angle_x = atan((W/2)/D)
      angle_y = atan((H/2)/D)
      max = max(angle_x, angle_y)
    """
    ax = math.degrees(math.atan((rect_w_m * 0.5) / distance_m))
    ay = math.degrees(math.atan((rect_h_m * 0.5) / distance_m))
    return float(max(ax, ay))

def main():
    cfg = load_json(CONFIG_JSON_PATH)

    optics = ensure_dict(cfg, "optics")
    camera = ensure_dict(cfg, "camera")
    laser = ensure_dict(cfg, "laser")
    calib = ensure_dict(cfg, "calibration")
    
    # =============================
    # Lecture des valeurs actuelles (JSON -> défaut)
    # =============================
    baseline_cur = float(optics.get("baseline_m", 0.10))
    n_photos_cur = int(calib.get("n_photos", 120))
    delay_cur = float(calib.get("countdown_s", 0.2))

    pattern_w_cur = int(calib.get("pattern_w", 9))
    pattern_h_cur = int(calib.get("pattern_h", 6))
    square_size_cur = float(calib.get("square_size_m", 0.025))

    cam_left_cur = int(camera.get("left_index", 0))
    cam_right_cur = int(camera.get("right_index", 1))

    # =============================
    # Inputs utilisateur (Entrée => garde valeur actuelle)
    # =============================
    baseline_m = input_float("Baseline camera-camera (m)", baseline_cur, min_v=0.001)
    n_photos = input_int("Nombre de paires de photos", n_photos_cur, min_v=10)
    delay_s = input_float("Secondes entre chaque capture", delay_cur, min_v=0.0)

    pattern_w = input_int("Damier: nb coins intérieurs (largeur)", pattern_w_cur, min_v=2)
    pattern_h = input_int("Damier: nb coins intérieurs (hauteur)", pattern_h_cur, min_v=2)
    square_size_m = input_float("Damier: taille d'une case (m)", square_size_cur, min_v=1e-6)

    cam_left = input_int("Index caméra gauche", cam_left_cur, min_v=0)
    cam_right = input_int("Index caméra droite", cam_right_cur, min_v=0)

    # =============================
    # Écriture dans le JSON
    # =============================
    optics["baseline_m"] = float(baseline_m)

    camera["left_index"] = int(cam_left)
    camera["right_index"] = int(cam_right)

    calib["n_photos"] = int(n_photos)
    calib["countdown_s"] = float(delay_s)
    calib["pattern_w"] = int(pattern_w)
    calib["pattern_h"] = int(pattern_h)
    calib["square_size_m"] = float(square_size_m)

    save_json(CONFIG_JSON_PATH, cfg)

    # ====== boucle capture + calib jusqu'à RMS OK ======
    while True:
        print("\n=== CAPTURE PHOTOS ===")
        launch_capture_ui(n_photos=n_photos, delay_s=delay_s)

        calib_dir = find_latest_calib_dir(DEFAULT_CALIB_ROOT)
        print(f"[INFO] Dossier capture utilisé: {calib_dir}")

        print("\n=== CALIBRATION STEREO ===")
        res = run_stereo_calib(
            calib_dir=calib_dir,
            pattern_w=pattern_w,
            pattern_h=pattern_h,
            square_size_m=square_size_m
        )

        rms = res["rms"]
        stereo_npz = res["stereo_npz"]

        print(f"[INFO] RMS = {rms} (seuil OK <= {RMS_OK_THRESHOLD})")
        if rms is None or rms > RMS_OK_THRESHOLD:
            print("[WARN] Calibration jugée mauvaise -> recommencer capture + calibration.")
            continue

        # OK -> on écrit dans JSON
        calib["calib_dir"] = calib_dir
        calib["stereo_npz_path"] = stereo_npz
        calib["stereo_rms"] = rms
        save_json(CONFIG_JSON_PATH, cfg)
        break

    # ====== alignment laser (aide) ======
    print("\n=== ALIGNMENT LASER (physique) ===")
    print("Une mire s'affiche + le flux cam gauche avec le centre en vert.")
    print("Ajuste mécaniquement le laser jusqu'à ce que le point laser tombe sur le centre caméra.")
    print("Quitter la mire: touche Q")
    laser_alignment_loop(cam_index_left=cam_left)

    # ====== mesure mur + rectangle pour calculer max_angle_deg ======
    print("\n=== MESURES POUR max_angle_deg ===")
    dist_m = input_float("Distance laser->mur (m)", laser.get("wall_distance_m", 3.0), min_v=0.01)
    rect_w_m = input_float("Largeur rectangle laser sur mur (m)", laser.get("rect_width_m", 1.0), min_v=0.001)
    rect_h_m = input_float("Hauteur rectangle laser sur mur (m)", laser.get("rect_height_m", 1.0), min_v=0.001)

    max_angle = compute_max_angle_deg(dist_m, rect_w_m, rect_h_m)

    laser["wall_distance_m"] = float(dist_m)
    laser["rect_width_m"] = float(rect_w_m)
    laser["rect_height_m"] = float(rect_h_m)
    laser["max_angle_deg"] = float(max_angle)

    save_json(CONFIG_JSON_PATH, cfg)

    print(f"\n[OK] max_angle_deg calculé = {max_angle:.3f} deg")
    print(f"[OK] JSON mis à jour: {CONFIG_JSON_PATH}")
    print(f"[OK] stereo_params.npz: {calib['stereo_npz_path']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Interrompu par utilisateur.")
        sys.exit(0)
