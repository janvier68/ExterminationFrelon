# calib_server_steps_full.py
# Flask multi-étapes Raspberry (sans GUI desktop)
# 1) Paramètres (préremplis depuis JSON) + h_cl_m + setup caméras + bouton swap (relance nécessaire)
# 2) Capture : flux live gauche/droite + séquence JS (countdown + /capture_photo) comme photoForCalib.py
# 3) Calibration : calcule matrice stéréo (mono + stereo + rectif + maps) comme stereo_calib.py
#    + contrôle RMS <= seuil, si NOK => nouvelle capture + recalib (boucle)
#    + affiche flux caméra gauche avec croix centre sur la page 3
# 4) Laser : page avec flux caméra gauche + croix centre + form distance mur/rectangle -> calc max_angle_deg
#    + écrit dans JSON + tirs test centre + côtés max (horizontal)

from flask import Flask, request, redirect, url_for, Response, render_template_string, jsonify
from markupsafe import Markup

from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import time
import os
import glob
import json
import math
import signal
import sys

import board
from Galvo import GalvoController

# ---------------------------
# Paths
# ---------------------------
CONFIG_PATH = "Config/config.json"
CALIB_ROOT_DIR = "Config/photos_calibration"
DEBUG_SUBDIR_NAME = "visualisation_calibration"

# ---------------------------
# Flask
# ---------------------------
app = Flask(__name__)

# ---------------------------
# State (simple)
# ---------------------------
STATE = {
    "calib_dir": None,
    "last_result": None,
    "last_laser": None,
}

# ---------------------------
# JSON helpers
# ---------------------------
def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: dict):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def ensure_dir_unique(base_dir: str) -> str:
    parent = os.path.dirname(base_dir)
    if parent:
        os.makedirs(parent, exist_ok=True)
    d = base_dir
    if os.path.exists(d):
        i = 1
        while os.path.exists(f"{base_dir}{i}"):
            i += 1
        d = f"{base_dir}{i}"
    os.makedirs(d, exist_ok=True)
    return d

def get_camera_indexes():
    cfg = load_json(CONFIG_PATH)
    cam = cfg.get("camera", {})
    left_i = int(cam.get("left_index", 1))
    right_i = int(cam.get("right_index", 0))
    return left_i, right_i

# ---------------------------
# Cameras (Picamera2) + MJPEG stream
# ---------------------------
LEFT_IDX, RIGHT_IDX = get_camera_indexes()

camL = Picamera2(camera_num=LEFT_IDX)
camR = Picamera2(camera_num=RIGHT_IDX)

cfgL = camL.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
cfgR = camR.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
camL.configure(cfgL)
camR.configure(cfgR)

camL.start()
camR.start()

lockL = threading.Lock()
lockR = threading.Lock()

def capture_bgr(camera: Picamera2, lock: threading.Lock) -> np.ndarray:
    with lock:
        req = camera.capture_request()
        frame = req.make_array("main")
        req.release()
    # Picamera2 -> RGB888 : convertir en BGR pour OpenCV
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def mjpeg_generator(which: str):
    cam = camL if which == "left" else camR
    lock = lockL if which == "left" else lockR
    while True:
        frame = capture_bgr(cam, lock)
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )

def draw_center_cross(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    L = 18
    t = 2
    cv2.line(frame_bgr, (cx - L, cy), (cx + L, cy), (255, 255, 255), t)
    cv2.line(frame_bgr, (cx, cy - L), (cx, cy + L), (255, 255, 255), t)
    cv2.circle(frame_bgr, (cx, cy), 3, (255, 255, 255), -1)
    return frame_bgr

def mjpeg_generator_left_center():
    while True:
        frame = capture_bgr(camL, lockL)
        frame = draw_center_cross(frame)
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )

# ---------------------------
# Calibration: version "stereo_calib.py"
# ---------------------------
def run_stereo_calibration(calib_dir: str, pattern_size: tuple[int, int], square_size_m: float):
    left_files = sorted(glob.glob(os.path.join(calib_dir, "left_*.png")))
    right_files = sorted(glob.glob(os.path.join(calib_dir, "right_*.png")))

    if len(left_files) == 0 or len(left_files) != len(right_files):
        raise RuntimeError("Nb d'images gauche/droite incohérent ou nul")

    debug_dir = os.path.join(calib_dir, DEBUG_SUBDIR_NAME)
    os.makedirs(debug_dir, exist_ok=True)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= float(square_size_m)

    objpoints, imgpointsL, imgpointsR = [], [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

    last_imgL = None
    used = 0

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
        last_imgL = imgL
        used += 1

        visL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
        visR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(visL, pattern_size, cornersL, True)
        cv2.drawChessboardCorners(visR, pattern_size, cornersR, True)
        concat = np.hstack((visL, visR))
        cv2.imwrite(os.path.join(debug_dir, f"pair_{idx:02d}.png"), concat)

    if used == 0 or last_imgL is None:
        raise RuntimeError("Aucun damier valide trouvé sur les paires d'images")

    h, w = last_imgL.shape[:2]
    image_size = (w, h)

    # mono calib
    _, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, image_size, None, None)
    _, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, image_size, None, None)

    # stereo calib
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = cv2.CALIB_FIX_INTRINSIC

    rms, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        mtxL, distL, mtxR, distR,
        image_size,
        criteria=criteria_stereo,
        flags=flags
    )

    # rectification + maps
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

    return float(rms), out_npz, debug_dir, int(used)

# ---------------------------
# Laser align
# ---------------------------
def gpio_from_string(pin: str):
    return getattr(board, pin)

def compute_max_angle_deg(distance_wall_m: float, rect_w_m: float, rect_h_m: float) -> float:
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

    galvo = GalvoController(
        max_angle_deg=max_angle,
        laser_pin=gpio_from_string(pin_str),
        gain=int(laser_cfg.get("gain", 1)),
        safe_start=bool(laser_cfg.get("safe_start", True)),
        max_code=int(laser_cfg.get("max_code", 4095)),
    )

    try:
        # centre
        galvo.set_angles(0.0, 0.0)
        time.sleep(0.2)
        galvo.laser_on()
        time.sleep(0.2)
        galvo.laser_off()

        # côtés max (horizontal)
        for sx in (max_angle, -max_angle):
            galvo.set_angles(sx, 0.0)
            time.sleep(0.2)
            galvo.laser_on()
            time.sleep(0.2)
            galvo.laser_off()
    finally:
        galvo.shutdown()

    return max_angle

# ---------------------------
# Defaults
# ---------------------------
def defaults():
    cfg = load_json(CONFIG_PATH)
    optics = cfg.get("optics", {})
    cap = cfg.get("calibration_capture", {})
    qual = cfg.get("calibration_quality", {})
    laser = cfg.get("laser", {})
    cam = cfg.get("camera", {})

    pattern = optics.get("pattern_size", [9, 6])
    if not (isinstance(pattern, (list, tuple)) and len(pattern) == 2):
        pattern = [9, 6]

    return {
        "baseline_m": float(optics.get("baseline_m", 0.10)),
        "h_cl_m": float(optics.get("h_cl_m", 0.0)),
        "n_photos": int(cap.get("n_photos", 120)),
        "countdown_s": float(cap.get("countdown_s", 0.2)),
        "pattern_cols": int(pattern[0]),
        "pattern_rows": int(pattern[1]),
        "square_size_m": float(optics.get("square_size_m", 0.025)),
        "rms_threshold": float(qual.get("rms_threshold", 0.8)),
        "distance_wall_m": float(laser.get("alignment", {}).get("distance_wall_m", 2.0)),
        "rect_w_m": float(laser.get("alignment", {}).get("rect_w_m", 1.0)),
        "rect_h_m": float(laser.get("alignment", {}).get("rect_h_m", 0.6)),
        "left_index": int(cam.get("left_index", 1)),
        "right_index": int(cam.get("right_index", 0)),
    }

# ---------------------------
# HTML
# ---------------------------
BASE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Calibration</title>
  <style>
    body{font-family:sans-serif;margin:24px;max-width:1200px}
    .row{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
    label{display:block;margin-top:10px}
    input{width:260px;padding:6px}
    button{padding:8px 14px;margin-top:14px}
    .card{border:1px solid #ddd;border-radius:12px;padding:14px;margin-top:16px}
    pre{background:#f6f6f6;padding:12px;border-radius:10px;overflow:auto}
    .nav a{margin-right:12px}
    img{border-radius:12px;border:1px solid #ddd}
  </style>
</head>
<body>
  <div class="nav">
    <a href="/step1">1) Paramètres</a>
    <a href="/step2">2) Capture</a>
    <a href="/step3">3) Calibration</a>
    <a href="/step4">4) Laser</a>
  </div>
  {{ body | safe }}
</body>
</html>
"""

STEP1 = """
<h2>1) Paramètres (préremplis depuis JSON)</h2>

<div class="card">
  <div><b>Setup caméras</b></div>
  <div class="row">
    <div>Index gauche: <b>{{left_index}}</b></div>
    <div>Index droite: <b>{{right_index}}</b></div>
  </div>
  <form method="POST" action="/swap_cameras">
    <button type="submit">Inverser les 2 caméras</button>
  </form>
  {% if cam_note %}
  <pre>{{cam_note}}</pre>
  {% endif %}
</div>

<form method="POST" action="/step1_save">
  <div class="row">
    <label>baseline (m)<br>
      <input name="baseline_m" type="number" step="0.001" value="{{baseline_m}}">
    </label>

    <label>h_cl_m (m)<br>
      <input name="h_cl_m" type="number" step="0.001" value="{{h_cl_m}}">
    </label>

    <label>N_PHOTOS (paires)<br>
      <input name="n_photos" type="number" step="1" value="{{n_photos}}">
    </label>

    <label>COUNTDOWN_SECONDS (s)<br>
      <input name="countdown_s" type="number" step="0.01" value="{{countdown_s}}">
    </label>

    <label>pattern cols<br>
      <input name="pattern_cols" type="number" step="1" value="{{pattern_cols}}">
    </label>

    <label>pattern rows<br>
      <input name="pattern_rows" type="number" step="1" value="{{pattern_rows}}">
    </label>

    <label>taille carré (m)<br>
      <input name="square_size_m" type="number" step="0.0001" value="{{square_size_m}}">
    </label>

    <label>seuil RMS (<=)<br>
      <input name="rms_threshold" type="number" step="0.0001" value="{{rms_threshold}}">
    </label>
  </div>

  <button type="submit">Enregistrer dans JSON</button>
</form>

{% if saved %}
<div class="card"><pre>{{saved}}</pre></div>
{% endif %}
"""

STEP2 = """
<h2>2) Capture (flux live + séquence)</h2>
<h1 style="color:red;">Vérifer caméra droit bien à droite sinon revenir avant et changer</h1>

<div class="card">
  <div><b>Dossier capture courant</b></div>
  <pre id="calib_dir">{{calib_dir}}</pre>
  <form method="POST" action="/new_capture_dir">
    <button type="submit">Nouveau dossier capture</button>
  </form>
</div>

<div class="card">
  <div class="row">
    <div style="flex:1">
      <div>Caméra gauche</div>
      <img src="/stream/left" width="480">
    </div>
    <div style="flex:1">
      <div>Caméra droite</div>
      <img src="/stream/right" width="480">
    </div>
  </div>
</div>

<div class="card">
  <div class="row" style="align-items:center">
    <div><b id="step_info"></b></div>
    <button id="start_btn" onclick="startSequence()">Lancer la séquence</button>
    <div id="counter" style="font-weight:bold"></div>
  </div>
  <div id="instruction" style="margin-top:8px"></div>
  <pre id="status" style="margin-top:10px"></pre>
</div>

<script>
  const N_PHOTOS = {{n_photos}};
  const COUNTDOWN_SECONDS = {{countdown_s}};
  const STEPS = Array.from({length: N_PHOTOS}, (_, i) =>
    `Photo ${i+1}/${N_PHOTOS} : déplace légèrement le damier puis attends la fin du compte à rebours.`
  );

  let currentStep = 0;
  let sequenceRunning = false;

  function q(sel){ return document.querySelector(sel); }
  function sleep(ms){ return new Promise(r => setTimeout(r, ms)); }

  function updateUI(){
    if (currentStep >= N_PHOTOS){
      q('#step_info').textContent = 'Toutes les photos sont prises.';
      q('#instruction').textContent = 'Séquence terminée.';
      q('#counter').textContent = '';
      q('#start_btn').disabled = false;
      return;
    }
    q('#step_info').textContent = `Photo ${currentStep+1} / ${N_PHOTOS}`;
    q('#instruction').textContent = STEPS[currentStep];
  }

  async function captureOneWithCountdown(){
    let remaining = Math.ceil(COUNTDOWN_SECONDS);
    if (COUNTDOWN_SECONDS <= 0) remaining = 0;

    if (remaining > 0){
      q('#counter').textContent = remaining + ' s';
      while (remaining > 0){
        await sleep(1000);
        remaining -= 1;
        q('#counter').textContent = remaining > 0 ? (remaining + ' s') : 'Capture...';
      }
    } else {
      q('#counter').textContent = 'Capture...';
    }

    try{
      const r = await fetch('/capture_photo', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({step_index: currentStep})
      });
      const d = await r.json();
      if (!d.ok){
        q('#status').textContent += `\\nErreur capture step ${currentStep}`;
        return false;
      }
      q('#status').textContent +=
        `\\nOK step ${currentStep}:\\n  ${d.left_file}\\n  ${d.right_file}\\n`;
      currentStep += 1;
      updateUI();
      return true;
    } catch(e){
      q('#status').textContent += `\\nErreur réseau step ${currentStep}: ${e}`;
      return false;
    } finally {
      q('#counter').textContent = '';
    }
  }

  async function startSequence(){
    if (sequenceRunning) return;
    sequenceRunning = true;
    currentStep = 0;
    q('#status').textContent = '';
    q('#start_btn').disabled = true;
    updateUI();

    while (currentStep < N_PHOTOS && sequenceRunning){
      const ok = await captureOneWithCountdown();
      if (!ok) break;
      await sleep(300);
    }

    sequenceRunning = false;
    q('#start_btn').disabled = false;
    updateUI();
  }

  window.addEventListener('load', updateUI);
</script>
"""

STEP3 = """
<h2>3) Calibration (RMS + boucle si NOK)</h2>

<div class="card">
  <div><b>Centre caméra gauche (overlay)</b></div>
  <img src="/stream/left_center" width="640">
</div>

<div class="card">
  <div>Dossier utilisé :</div>
  <pre>{{calib_dir}}</pre>

  <form method="POST" action="/run_calibration_loop">
    <label>Max tentatives<br>
      <input name="max_attempts" type="number" step="1" value="10">
    </label><br>
    <button type="submit">Lancer Calibration (re-capture si NOK)</button>
  </form>
</div>

{% if result %}
<div class="card"><pre>{{result}}</pre></div>
{% endif %}
"""

STEP4 = """
<h2>4) Alignement laser</h2>

<div class="card">
  <div><b>Caméra gauche + centre</b></div>
  <img src="/stream/left_center" width="640">
</div>

<form method="POST" action="/run_laser_align">
  <div class="row">
    <label>distance mur (m)<br>
      <input name="distance_wall_m" type="number" step="0.01" value="{{distance_wall_m}}">
    </label>
    <label>largeur rectangle laser (m)<br>
      <input name="rect_w_m" type="number" step="0.01" value="{{rect_w_m}}">
    </label>
    <label>hauteur rectangle laser (m)<br>
      <input name="rect_h_m" type="number" step="0.01" value="{{rect_h_m}}">
    </label>
  </div>
  <button type="submit">Calculer max_angle_deg + tirs (centre + côtés)</button>
</form>

{% if laser %}
<div class="card"><pre>{{laser}}</pre></div>
{% endif %}
"""

# ---------------------------
# Routes: streams
# ---------------------------
@app.get("/stream/<which>")
def stream(which: str):
    if which not in ("left", "right"):
        return "bad stream", 400
    return Response(mjpeg_generator(which), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream/left_center")
def stream_left_center():
    return Response(mjpeg_generator_left_center(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ---------------------------
# Routes: navigation
# ---------------------------
@app.get("/")
def root():
    return redirect(url_for("step1"))

# ---------------------------
# Step 1
# ---------------------------
@app.get("/step1")
def step1():
    d = defaults()
    body_html = render_template_string(STEP1, **d, saved=None, cam_note=None)
    return render_template_string(BASE, body=Markup(body_html))

@app.post("/step1_save")
def step1_save():
    baseline_m = float(request.form["baseline_m"])
    h_cl_m = float(request.form["h_cl_m"])
    n_photos = int(request.form["n_photos"])
    countdown_s = float(request.form["countdown_s"])
    pattern_cols = int(request.form["pattern_cols"])
    pattern_rows = int(request.form["pattern_rows"])
    square_size_m = float(request.form["square_size_m"])
    rms_threshold = float(request.form["rms_threshold"])

    cfg = load_json(CONFIG_PATH)

    cfg.setdefault("optics", {})
    cfg["optics"]["baseline_m"] = baseline_m
    cfg["optics"]["h_cl_m"] = h_cl_m
    cfg["optics"]["pattern_size"] = [pattern_cols, pattern_rows]
    cfg["optics"]["square_size_m"] = square_size_m

    cfg.setdefault("calibration_capture", {})
    cfg["calibration_capture"]["n_photos"] = n_photos
    cfg["calibration_capture"]["countdown_s"] = countdown_s

    cfg.setdefault("calibration_quality", {})
    cfg["calibration_quality"]["rms_threshold"] = rms_threshold

    save_json(CONFIG_PATH, cfg)

    d2 = defaults()
    body_html = render_template_string(STEP1, **d2, saved=json.dumps({"saved": True}, indent=2), cam_note=None)
    return render_template_string(BASE, body=Markup(body_html))

@app.post("/swap_cameras")
def swap_cameras():
    cfg = load_json(CONFIG_PATH)
    cam = cfg.setdefault("camera", {})
    left_i = int(cam.get("left_index", 1))
    right_i = int(cam.get("right_index", 0))
    cam["left_index"], cam["right_index"] = right_i, left_i
    save_json(CONFIG_PATH, cfg)

    d = defaults()
    note = "Caméras inversées dans le JSON. Relance le script pour que ça s'applique (Picamera2 est initialisée au démarrage)."
    body_html = render_template_string(STEP1, **d, saved=None, cam_note=note)
    return render_template_string(BASE, body=Markup(body_html))

# ---------------------------
# Step 2 (capture)
# ---------------------------
@app.post("/new_capture_dir")
def new_capture_dir():
    STATE["calib_dir"] = ensure_dir_unique(CALIB_ROOT_DIR)
    return redirect(url_for("step2"))

@app.get("/step2")
def step2():
    d = defaults()
    if not STATE["calib_dir"]:
        STATE["calib_dir"] = ensure_dir_unique(CALIB_ROOT_DIR)
    body_html = render_template_string(STEP2, **d, calib_dir=STATE["calib_dir"])
    return render_template_string(BASE, body=Markup(body_html))

@app.post("/capture_photo")
def capture_photo():
    data = request.get_json(force=True)
    step_index = int(data.get("step_index", 0))

    if not STATE["calib_dir"]:
        STATE["calib_dir"] = ensure_dir_unique(CALIB_ROOT_DIR)
    calib_dir = STATE["calib_dir"]

    img_left = capture_bgr(camL, lockL)
    img_right = capture_bgr(camR, lockR)

    fname_left = os.path.join(calib_dir, f"left_{step_index:03d}.png")
    fname_right = os.path.join(calib_dir, f"right_{step_index:03d}.png")

    okL = cv2.imwrite(fname_left, img_left)
    okR = cv2.imwrite(fname_right, img_right)

    return jsonify({
        "ok": bool(okL and okR),
        "step_index": step_index,
        "left_file": fname_left,
        "right_file": fname_right,
        "calib_dir": calib_dir
    })

# ---------------------------
# Step 3 (calibration loop)
# ---------------------------
@app.get("/step3")
def step3():
    d = defaults()
    calib_dir = STATE["calib_dir"] or "(capture d'abord)"
    body_html = render_template_string(STEP3, **d, calib_dir=calib_dir, result=STATE["last_result"])
    return render_template_string(BASE, body=Markup(body_html))

def capture_burst(calib_dir: str, n_photos: int, countdown_s: float):
    # fallback automatique utilisé par la boucle de calibration en cas de NOK
    for i in range(n_photos):
        if countdown_s > 0:
            time.sleep(float(countdown_s))
        imgL = capture_bgr(camL, lockL)
        imgR = capture_bgr(camR, lockR)
        cv2.imwrite(os.path.join(calib_dir, f"left_{i:03d}.png"), imgL)
        cv2.imwrite(os.path.join(calib_dir, f"right_{i:03d}.png"), imgR)

@app.post("/run_calibration_loop")
def run_calibration_loop():
    d = defaults()
    max_attempts = int(request.form.get("max_attempts", "10"))

    pattern_size = (d["pattern_cols"], d["pattern_rows"])
    rms_threshold = float(d["rms_threshold"])

    last = None

    for attempt in range(1, max_attempts + 1):
        if not STATE["calib_dir"]:
            STATE["calib_dir"] = ensure_dir_unique(CALIB_ROOT_DIR)
            capture_burst(STATE["calib_dir"], d["n_photos"], d["countdown_s"])

        calib_dir = STATE["calib_dir"]

        try:
            rms, npz_path, debug_dir, used = run_stereo_calibration(calib_dir, pattern_size, d["square_size_m"])
            ok = rms <= rms_threshold
            last = {
                "attempt": attempt,
                "ok": ok,
                "rms": float(rms),
                "rms_threshold": float(rms_threshold),
                "used_pairs": int(used),
                "calib_dir": calib_dir,
                "debug_dir": debug_dir,
                "npz_path": npz_path,
            }
        except Exception as e:
            last = {
                "attempt": attempt,
                "ok": False,
                "error": str(e),
                "calib_dir": calib_dir,
            }
            ok = False

        if ok:
            cfg = load_json(CONFIG_PATH)
            cfg.setdefault("optics", {})
            cfg["optics"]["baseline_m"] = float(d["baseline_m"])
            cfg["optics"]["h_cl_m"] = float(d["h_cl_m"])
            cfg["optics"]["stereo_calibration_path"] = last["npz_path"]
            cfg["optics"]["calibration_photos_dir"] = last["calib_dir"]
            cfg["optics"]["pattern_size"] = [d["pattern_cols"], d["pattern_rows"]]
            cfg["optics"]["square_size_m"] = float(d["square_size_m"])
            cfg.setdefault("calibration_quality", {})
            cfg["calibration_quality"]["stereo_rms"] = float(last["rms"])
            save_json(CONFIG_PATH, cfg)

            STATE["last_result"] = json.dumps(last, indent=2, ensure_ascii=False)
            break

        # NOK => recapture auto dans un nouveau dossier
        STATE["calib_dir"] = ensure_dir_unique(CALIB_ROOT_DIR)
        capture_burst(STATE["calib_dir"], d["n_photos"], d["countdown_s"])
        STATE["last_result"] = json.dumps(last, indent=2, ensure_ascii=False)

    d2 = defaults()
    body_html = render_template_string(STEP3, **d2, calib_dir=STATE["calib_dir"] or "(?)", result=STATE["last_result"])
    return render_template_string(BASE, body=Markup(body_html))

# ---------------------------
# Step 4 (laser)
# ---------------------------
@app.get("/step4")
def step4():
    d = defaults()
    body_html = render_template_string(STEP4, **d, laser=STATE["last_laser"])
    return render_template_string(BASE, body=Markup(body_html))

@app.post("/run_laser_align")
def run_laser_align():
    distance_wall_m = float(request.form["distance_wall_m"])
    rect_w_m = float(request.form["rect_w_m"])
    rect_h_m = float(request.form["rect_h_m"])

    max_angle = laser_alignment_and_save(distance_wall_m, rect_w_m, rect_h_m)

    STATE["last_laser"] = json.dumps(
        {"max_angle_deg": float(max_angle), "json_written": CONFIG_PATH},
        indent=2,
        ensure_ascii=False
    )

    d = defaults()
    body_html = render_template_string(STEP4, **d, laser=STATE["last_laser"])
    return render_template_string(BASE, body=Markup(body_html))

# ---------------------------
# Shutdown propre
# ---------------------------
def shutdown(sig, frame):
    try:
        camL.close()
        camR.close()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

if __name__ == "__main__":
    print("http://0.0.0.0:5000/step1")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
