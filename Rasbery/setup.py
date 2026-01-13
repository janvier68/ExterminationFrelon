# calib_server.py
# Flask (web) uniquement : saisie paramètres -> capture paires -> calibration stéréo -> contrôle RMS -> boucle si NOK
# + Alignement laser : mesure rectangle sur mur -> calcule max_angle_deg -> écrit JSON

from flask import Flask, request, jsonify, render_template_string
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import time
import os
import glob
import json
import signal
import sys
import math

import board
from galvo import GalvoController  # même usage que main.py :contentReference[oaicite:0]{index=0}

# ---------------------------
# Fichiers
# ---------------------------
CONFIG_PATH = "Config/config.json"  # adapte si besoin
CALIB_ROOT_DIR = "Config/photos_calibration"
DEBUG_SUBDIR = "Config/visualisation_calibration"

# ---------------------------
# Flask
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Caméras (Picamera2)
# (même logique capture que photoForCalib.py) :contentReference[oaicite:1]{index=1}
# ---------------------------
camL = Picamera2(camera_num=1)  # adapte si inversé
camR = Picamera2(camera_num=0)
camL.start()
camR.start()
lockL = threading.Lock()
lockR = threading.Lock()

def capture_bgr(camera: Picamera2, lock: threading.Lock):
    with lock:
        req = camera.capture_request()
        frame = req.make_array("main")
        req.release()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# ---------------------------
# Config JSON helpers
# ---------------------------
def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def ensure_dir_unique(base_dir: str) -> str:
    d = base_dir
    if os.path.exists(d):
        i = 1
        while os.path.exists(f"{base_dir}{i}"):
            i += 1
        d = f"{base_dir}{i}"
    os.makedirs(d, exist_ok=True)
    return d

# ---------------------------
# Calibration stéréo
# (reprend la logique de calibrationCamera.py) :contentReference[oaicite:2]{index=2}
# ---------------------------
def run_stereo_calibration(calib_dir: str, pattern_size: tuple[int, int], square_size_m: float):
    left_glob = os.path.join(calib_dir, "left_*.png")
    right_glob = os.path.join(calib_dir, "right_*.png")
    left_files = sorted(glob.glob(left_glob))
    right_files = sorted(glob.glob(right_glob))

    if len(left_files) == 0 or len(left_files) != len(right_files):
        raise RuntimeError("Nb d'images gauche/droite incohérent ou nul")

    debug_dir = os.path.join(calib_dir, DEBUG_SUBDIR)
    os.makedirs(debug_dir, exist_ok=True)

    # points 3D du damier
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= float(square_size_m)

    objpoints = []
    imgpointsL = []
    imgpointsR = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

    imgL_last = None
    for idx, (lf, rf) in enumerate(zip(left_files, right_files)):
        imgL = cv2.imread(lf, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
        if imgL is None or imgR is None:
            continue

        retL, cornersL = cv2.findChessboardCorners(imgL, pattern_size)
        retR, cornersR = cv2.findChessboardCorners(imgR, pattern_size)
        if not (retL and retR):
            continue

        cornersL = cv2.cornerSubPix(imgL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(imgR, cornersR, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
        imgL_last = imgL

        # visu
        visL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
        visR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(visL, pattern_size, cornersL, True)
        cv2.drawChessboardCorners(visR, pattern_size, cornersR, True)
        cv2.imwrite(os.path.join(debug_dir, f"pair_{idx:03d}.png"), np.hstack((visL, visR)))

    if len(objpoints) == 0:
        raise RuntimeError("Aucun damier valide trouvé sur les paires d'images")

    h, w = imgL_last.shape[:2]
    image_size = (w, h)

    # mono calib
    _, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, image_size, None, None)
    _, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, image_size, None, None)

    # stereo calib (RMS = retS)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = cv2.CALIB_FIX_INTRINSIC

    retS, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        mtxL, distL, mtxR, distR,
        image_size,
        criteria=criteria_stereo,
        flags=flags
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtxL, distL, mtxR, distR,
        image_size, R, T, alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, image_size, cv2.CV_32FC1)

    out_npz = os.path.join(debug_dir, "stereo_params.npz")
    np.savez(
        out_npz,
        mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
        image_size=image_size
    )

    return float(retS), out_npz, debug_dir

# ---------------------------
# Capture + calibration en boucle
# ---------------------------
def capture_pairs(calib_dir: str, n_photos: int, countdown_s: float):
    for i in range(n_photos):
        if countdown_s > 0:
            time.sleep(float(countdown_s))

        imgL = capture_bgr(camL, lockL)
        imgR = capture_bgr(camR, lockR)

        cv2.imwrite(os.path.join(calib_dir, f"left_{i:03d}.png"), imgL)
        cv2.imwrite(os.path.join(calib_dir, f"right_{i:03d}.png"), imgR)

def capture_and_calibrate_loop(
    baseline_m: float,
    n_photos: int,
    countdown_s: float,
    pattern_cols: int,
    pattern_rows: int,
    square_size_m: float,
    rms_threshold: float,
    max_attempts: int = 10,
):
    pattern_size = (int(pattern_cols), int(pattern_rows))

    last = {}
    for attempt in range(1, max_attempts + 1):
        calib_dir = ensure_dir_unique(CALIB_ROOT_DIR)
        capture_pairs(calib_dir, int(n_photos), float(countdown_s))

        rms, npz_path, debug_dir = run_stereo_calibration(
            calib_dir=calib_dir,
            pattern_size=pattern_size,
            square_size_m=float(square_size_m),
        )

        last = {
            "attempt": attempt,
            "calib_dir": calib_dir,
            "debug_dir": debug_dir,
            "npz_path": npz_path,
            "rms": rms,
            "ok": rms <= float(rms_threshold),
        }

        if last["ok"]:
            # écrit JSON
            cfg = load_json(CONFIG_PATH)
            cfg.setdefault("optics", {})
            cfg["optics"]["baseline_m"] = float(baseline_m)
            cfg["optics"]["stereo_calibration_path"] = npz_path
            cfg["optics"]["calibration_photos_dir"] = calib_dir
            cfg["optics"]["pattern_size"] = [int(pattern_cols), int(pattern_rows)]
            cfg["optics"]["square_size_m"] = float(square_size_m)
            cfg.setdefault("calibration_quality", {})
            cfg["calibration_quality"]["stereo_rms"] = float(rms)
            save_json(CONFIG_PATH, cfg)
            return last

    return last  # NOK après max_attempts

# ---------------------------
# Laser align: calc max_angle_deg + écriture JSON + test tirs
# ---------------------------
def gpio_from_string(pin: str):
    return getattr(board, pin)

def compute_max_angle_deg(distance_wall_m: float, rect_w_m: float, rect_h_m: float) -> float:
    # demi-largeur / demi-hauteur -> angles max (petits angles ok, mais on calc exact)
    ax = math.degrees(math.atan((rect_w_m / 2.0) / distance_wall_m))
    ay = math.degrees(math.atan((rect_h_m / 2.0) / distance_wall_m))
    return float(max(ax, ay))

def laser_alignment_and_save(distance_wall_m: float, rect_w_m: float, rect_h_m: float):
    cfg = load_json(CONFIG_PATH)
    laser_cfg = cfg.setdefault("laser", {})
    pin_str = laser_cfg.get("laser_pin", "D17")

    max_angle = compute_max_angle_deg(float(distance_wall_m), float(rect_w_m), float(rect_h_m))
    laser_cfg["max_angle_deg"] = max_angle
    laser_cfg["alignment"] = {
        "distance_wall_m": float(distance_wall_m),
        "rect_w_m": float(rect_w_m),
        "rect_h_m": float(rect_h_m),
    }

    save_json(CONFIG_PATH, cfg)

    # test tir: centre + côté max (comme demandé)
    galvo = GalvoController(
        max_angle_deg=max_angle,
        laser_pin=gpio_from_string(pin_str),
        gain=int(laser_cfg.get("gain", 2)),
        safe_start=bool(laser_cfg.get("safe_start", True)),
    )

    try:
        # centre
        galvo.set_angles(theta_x=0.0, theta_y=0.0)
        time.sleep(0.2)
        galvo.laser_on()
        time.sleep(0.2)
        galvo.laser_off()

        # côté max horizontal (droite)
        galvo.set_angles(theta_x=max_angle, theta_y=0.0)
        time.sleep(0.2)
        galvo.laser_on()
        time.sleep(0.2)
        galvo.laser_off()

        # côté max horizontal (gauche)
        galvo.set_angles(theta_x=-max_angle, theta_y=0.0)
        time.sleep(0.2)
        galvo.laser_on()
        time.sleep(0.2)
        galvo.laser_off()
    finally:
        galvo.shutdown()

    return max_angle

# ---------------------------
# UI minimal (form préremplie depuis JSON)
# ---------------------------
HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Calibration & Laser</title>
    <style>
      body { font-family: sans-serif; margin: 24px; max-width: 820px; }
      label { display:block; margin-top: 12px; }
      input { width: 260px; padding: 6px; }
      button { margin-top: 16px; padding: 8px 14px; }
      .row { display:flex; gap:18px; flex-wrap: wrap; }
      .card { border:1px solid #ddd; border-radius:10px; padding:16px; margin-top:16px; }
      pre { background:#f6f6f6; padding:12px; border-radius:10px; overflow:auto; }
    </style>
  </head>
  <body>
    <h2>Calibration stéréo</h2>
    <form method="POST" action="/run_calibration">
      <div class="row">
        <label>baseline (m)
          <input name="baseline_m" type="number" step="0.001" value="{{baseline_m}}">
        </label>

        <label>nb photos (paires)
          <input name="n_photos" type="number" step="1" value="{{n_photos}}">
        </label>

        <label>délai / COUNTDOWN (s)
          <input name="countdown_s" type="number" step="0.01" value="{{countdown_s}}">
        </label>

        <label>pattern damier (cols)
          <input name="pattern_cols" type="number" step="1" value="{{pattern_cols}}">
        </label>

        <label>pattern damier (rows)
          <input name="pattern_rows" type="number" step="1" value="{{pattern_rows}}">
        </label>

        <label>taille carré (m)
          <input name="square_size_m" type="number" step="0.0001" value="{{square_size_m}}">
        </label>

        <label>seuil RMS (<=)
          <input name="rms_threshold" type="number" step="0.0001" value="{{rms_threshold}}">
        </label>
      </div>

      <button type="submit">Capture + Calibration (boucle si NOK)</button>
    </form>

    {% if result %}
      <div class="card">
        <h3>Résultat</h3>
        <pre>{{result}}</pre>
      </div>
    {% endif %}

    {% if laser_result %}
      <div class="card">
        <h3>Laser</h3>
        <pre>{{laser_result}}</pre>
      </div>
    {% endif %}
  </body>
</html>
"""

# <h2>Alignement laser</h2>
# <form method="POST" action="/run_laser_align">
#   <div class="row">
#     <label>distance mur (m)
#       <input name="distance_wall_m" type="number" step="0.01" value="{{distance_wall_m}}">
#     </label>

#     <label>largeur rectangle laser (m)
#       <input name="rect_w_m" type="number" step="0.01" value="{{rect_w_m}}">
#     </label>

#     <label>hauteur rectangle laser (m)
#       <input name="rect_h_m" type="number" step="0.01" value="{{rect_h_m}}">
#     </label>
#   </div>
#   <button type="submit">Calcul max_angle + tirs (centre + côtés)</button>
# </form>

def get_defaults_from_json():
    cfg = load_json(CONFIG_PATH)

    optics = cfg.get("optics", {})
    laser = cfg.get("laser", {})

    pattern = optics.get("pattern_size", [9, 6])
    if isinstance(pattern, (list, tuple)) and len(pattern) == 2:
        pattern_cols, pattern_rows = int(pattern[0]), int(pattern[1])
    else:
        pattern_cols, pattern_rows = 9, 6

    return {
        "baseline_m": float(optics.get("baseline_m", 0.10)),
        "n_photos": int(cfg.get("calibration_capture", {}).get("n_photos", 120)),
        "countdown_s": float(cfg.get("calibration_capture", {}).get("countdown_s", 0.2)),
        "pattern_cols": pattern_cols,
        "pattern_rows": pattern_rows,
        "square_size_m": float(optics.get("square_size_m", 0.025)),
        "rms_threshold": float(cfg.get("calibration_quality", {}).get("rms_threshold", 0.8)),
        "distance_wall_m": float(laser.get("alignment", {}).get("distance_wall_m", 2.0)),
        "rect_w_m": float(laser.get("alignment", {}).get("rect_w_m", 1.0)),
        "rect_h_m": float(laser.get("alignment", {}).get("rect_h_m", 0.6)),
    }

@app.route("/", methods=["GET"])
def index():
    d = get_defaults_from_json()
    return render_template_string(HTML, **d, result=None, laser_result=None)

@app.route("/run_calibration", methods=["POST"])
def run_calibration():
    baseline_m = float(request.form["baseline_m"])
    n_photos = int(request.form["n_photos"])
    countdown_s = float(request.form["countdown_s"])
    pattern_cols = int(request.form["pattern_cols"])
    pattern_rows = int(request.form["pattern_rows"])
    square_size_m = float(request.form["square_size_m"])
    rms_threshold = float(request.form["rms_threshold"])

    # stocke aussi dans JSON les paramètres de capture
    cfg = load_json(CONFIG_PATH)
    cfg.setdefault("calibration_capture", {})
    cfg["calibration_capture"]["n_photos"] = n_photos
    cfg["calibration_capture"]["countdown_s"] = countdown_s
    cfg.setdefault("calibration_quality", {})
    cfg["calibration_quality"]["rms_threshold"] = rms_threshold
    save_json(CONFIG_PATH, cfg)

    result = capture_and_calibrate_loop(
        baseline_m=baseline_m,
        n_photos=n_photos,
        countdown_s=countdown_s,
        pattern_cols=pattern_cols,
        pattern_rows=pattern_rows,
        square_size_m=square_size_m,
        rms_threshold=rms_threshold,
    )

    d = get_defaults_from_json()
    return render_template_string(HTML, **d, result=json.dumps(result, indent=2, ensure_ascii=False), laser_result=None)

@app.route("/run_laser_align", methods=["POST"])
def run_laser_align():
    distance_wall_m = float(request.form["distance_wall_m"])
    rect_w_m = float(request.form["rect_w_m"])
    rect_h_m = float(request.form["rect_h_m"])

    max_angle = laser_alignment_and_save(distance_wall_m, rect_w_m, rect_h_m)

    d = get_defaults_from_json()
    laser_result = {
        "max_angle_deg_written_to_json": max_angle,
        "json_path": CONFIG_PATH
    }
    return render_template_string(HTML, **d, result=None, laser_result=json.dumps(laser_result, indent=2, ensure_ascii=False))

# ---------------------------
# Shutdown propre
# ---------------------------
def signal_handler(sig, frame):
    try:
        camL.close()
        camR.close()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
