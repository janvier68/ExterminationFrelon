from flask import Flask, Response, render_template_string, request, jsonify
from picamera2 import Picamera2
# from ..config import N_PHOTOS, COUNTDOWN_SECONDS
import cv2
import numpy as np
import threading
import signal
import sys
import time
import os
import json

app = Flask(__name__)

# ---------- Config ----------
CALIB_DIR = "photos_calibration"
# Si "calib" existe déjà, créer "calibX" avec X croissant
if os.path.exists(CALIB_DIR):
    i = 1
    new_dir = f"{CALIB_DIR}{i}"
    while os.path.exists(new_dir):
        i += 1
        new_dir = f"{CALIB_DIR}{i}"
    CALIB_DIR = new_dir

os.makedirs(CALIB_DIR, exist_ok=True)
# Nombre total de photos (paires gauche/droite)
N_PHOTOS = 120

# Délai en secondes avant chaque capture
COUNTDOWN_SECONDS = 0.2

# Génération d'instructions pour chaque photo
BOARD_STEPS = [
    f"Photo {i+1}/{N_PHOTOS} : déplace légèrement le board par rapport à la précédente, "
    f"puis attends la fin du compte à rebours."
    for i in range(N_PHOTOS)
]
BOARD_STEPS_JS = json.dumps(BOARD_STEPS, ensure_ascii=False)
# ---------- Caméras ----------
cam0 = Picamera2(camera_num=1)
cam1 = Picamera2(camera_num=0)
cam0.start()
cam1.start()

lock0 = threading.Lock()
lock1 = threading.Lock()


def capture_bgr(camera, lock):
    """Capture une image BGR depuis une Picamera2."""
    with lock:
        req = camera.capture_request()
        frame = req.make_array("main")
        req.release()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def generate_frames(which="left"):
    """Flux MJPEG pour prévisualisation."""
    while True:
        if which == "left":
            frame = capture_bgr(cam0, lock0)
        else:
            frame = capture_bgr(cam1, lock1)
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buf.tobytes() + b"\r\n"
        )

# ---------- Routes vidéo ----------
@app.route("/video_feed0")
def video_feed0():
    return Response(
        generate_frames("left"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/video_feed1")
def video_feed1():
    return Response(
        generate_frames("right"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ---------- Capture d'une paire de photos ----------
@app.route("/capture_photo", methods=["POST"])
def capture_photo():
    """
    JSON attendu:
      { "step_index": int }
    Sauvegarde:
      calib/left_XX.png
      calib/right_XX.png
    """
    data = request.get_json(force=True)
    step_index = int(data.get("step_index", 0))

    img_left = capture_bgr(cam0, lock0)
    img_right = capture_bgr(cam1, lock1)

    fname_left = os.path.join(CALIB_DIR, f"left_{step_index:02d}.png")
    fname_right = os.path.join(CALIB_DIR, f"right_{step_index:02d}.png")

    cv2.imwrite(fname_left, img_left)
    cv2.imwrite(fname_right, img_right)

    print(f"[CAPTURE] step {step_index:02d} -> {fname_left}, {fname_right}")

    return jsonify({
        "ok": True,
        "step_index": step_index,
        "left_file": fname_left,
        "right_file": fname_right
    })

# ---------- Page principale ----------
@app.route("/")
def index():
    html = f"""
    <html>
    <head>
      <title>Capture board calibration</title>
      <style>
        body {{
          background:#111; color:#eee; font-family:sans-serif;
          margin:0; padding:20px;
        }}
        .row {{ display:flex; gap:16px; align-items:flex-start; }}
        .col {{ flex:1; }}
        img.stream {{
          width:100%; border:2px solid #444; border-radius:4px;
        }}
        .toolbar {{
          display:flex; gap:16px; align-items:center; margin-bottom:16px;
        }}
        .badge {{
          padding:4px 8px; background:#222; border:1px solid #444;
          border-radius:6px;
        }}
        button {{
          background:#2d6cdf; color:#fff; border:none;
          padding:8px 16px; border-radius:6px; font-size:15px;
        }}
        button:disabled {{ background:#555; }}
        #instruction {{ font-size:16px; margin-top:8px; }}
        #counter {{ font-size:22px; font-weight:bold; margin-left:8px; }}
        #status {{ margin-top:10px; color:#9fe870; white-space:pre-line; }}
      </style>
      <script>
        const STEPS = {BOARD_STEPS_JS};
        const TOTAL_STEPS = STEPS.length;
        const COUNTDOWN_SECONDS = {COUNTDOWN_SECONDS};

        let currentStep = 0;
        let countdownTimer = null;
        let sequenceRunning = false;

        function q(sel) {{ return document.querySelector(sel); }}

        function updateUI() {{
          if (currentStep >= TOTAL_STEPS) {{
            q('#step_info').textContent = 'Toutes les photos sont prises.';
            q('#instruction').textContent = 'Séquence terminée.';
            q('#start_btn').disabled = false;
            q('#counter').textContent = '';
          }} else {{
            q('#step_info').textContent =
              'Photo ' + (currentStep+1) + ' / ' + TOTAL_STEPS;
            q('#instruction').textContent = STEPS[currentStep];
          }}
        }}

        function sleep(ms) {{
          return new Promise(resolve => setTimeout(resolve, ms));
        }}

        async function captureOneWithCountdown() {{
          let remaining = COUNTDOWN_SECONDS;
          q('#counter').textContent = remaining + ' s';

          // Compte à rebours
          while (remaining > 0) {{
            await sleep(1000);
            remaining -= 1;
            if (remaining > 0) {{
              q('#counter').textContent = remaining + ' s';
            }} else {{
              q('#counter').textContent = 'Capture...';
            }}
          }}

          // Requête de capture
          try {{
            const r = await fetch('/capture_photo', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify({{ step_index: currentStep }})
            }});
            const d = await r.json();

            if (d.ok) {{
              const msg =
                'Photo ' + (currentStep+1) + ' sauvegardée:\\n' +
                '  ' + d.left_file + '\\n' +
                '  ' + d.right_file;
              const prev = q('#status').textContent;
              q('#status').textContent = (prev ? prev + '\\n' : '') + msg;

              currentStep += 1;
              updateUI();
            }} else {{
              q('#status').textContent += '\\nErreur capture (step ' +
                (currentStep+1) + ').';
              sequenceRunning = false;
              return false;
            }}
          }} catch (e) {{
            console.error(e);
            q('#status').textContent += '\\nErreur réseau (step ' +
              (currentStep+1) + ').';
            sequenceRunning = false;
            return false;
          }} finally {{
            q('#counter').textContent = '';
          }}
          return true;
        }}

        async function startSequence() {{
          if (sequenceRunning) return;
          sequenceRunning = true;
          currentStep = 0;
          q('#status').textContent = '';
          q('#start_btn').disabled = true;
          updateUI();

          while (currentStep < TOTAL_STEPS && sequenceRunning) {{
            const ok = await captureOneWithCountdown();
            if (!ok) break;
            // Petite pause entre les captures si tu veux
            await sleep(500);
          }}

          sequenceRunning = false;
          q('#start_btn').disabled = false;
          if (currentStep >= TOTAL_STEPS) {{
            q('#step_info').textContent = 'Toutes les photos sont prises.';
            q('#instruction').textContent = 'Séquence terminée.';
          }}
        }}

        window.addEventListener('load', () => {{
          updateUI();
        }});
      </script>
    </head>
    <body>
      <div class="toolbar">
        <span class="badge" id="step_info"></span>
        <button id="start_btn" onclick="startSequence()">
          Lancer la séquence ({N_PHOTOS} photos, délai {COUNTDOWN_SECONDS} s)
        </button>
        <span id="counter"></span>
      </div>

      <div id="instruction"></div>

      <div class="row" style="margin-top:16px;">
        <div class="col">
          <div class="badge">Caméra 0 (gauche)</div>
          <img class="stream" src="/video_feed0">
        </div>
        <div class="col">
          <div class="badge">Caméra 1 (droite)</div>
          <img class="stream" src="/video_feed1">
        </div>
      </div>

      <div id="status"></div>
    </body>
    </html>
    """
    return render_template_string(html)

# ---------- Gestion ctrl+C ----------
def signal_handler(sig, frame):
    cam0.close()
    cam1.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("[INFO] Calibration capture UI: http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)