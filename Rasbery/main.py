from flask import Flask, Response, render_template_string
import signal
import sys
import time
import threading
import board 

from Galvo import GalvoController
from Cam import CameraManager
from Profondeur import StereoAngleCalculator
from Config import load_config,gpio_from_string

app = Flask(__name__)

# =============================
# paramètres
# =============================
cfg = load_config("config/config.json")

SCORE_MIN_DETECTION = float(cfg["detection"]["score_min_detection"])
PAIR_MAX_DIST_PX = int(cfg["detection"]["pair_max_dist_px"])
TEMPS_LASER_SHOOT = float(cfg["laser"]["shoot_time_s"])
BORDER_SEC = float(cfg["security"]["border_sec"])

# =============================
# Initialisation galvo
# =============================
galvo = GalvoController(
    max_angle_deg=float(cfg["laser"]["max_angle_deg"]),
    laser_pin=gpio_from_string(cfg["laser"]["laser_pin"]),
    gain=int(cfg["laser"]["gain"]),
    safe_start=bool(cfg["laser"].get("safe_start", True)),
)

# =============================
# Profondeur + angles
# =============================
calc = StereoAngleCalculator(
    baseline_m=float(cfg["optics"]["baseline_m"]),
    h_cl_m=float(cfg["optics"]["h_cl_m"]),
)

calib_path = cfg["optics"].get("stereo_calibration_path")
try:
    calc.load_stereo_calibration(calib_path)
except Exception as e:
    raise RuntimeError("Impossible de charger le fichier de calibration") from e

# =============================
# Caméras
# =============================
camG = CameraManager(pathZim=str(cfg["camera"]["ia_packerOutZip_path"]),pathLab=str(cfg["camera"]["ia_labelTxt_path"]),cam_index=int(cfg["camera"]["left_index"]), safeZone=float(cfg["security"]["border_sec"]),frame_rate=int(cfg["camera"]["frame_rate"]))   # gauche
camD = CameraManager(pathZim=str(cfg["camera"]["ia_packerOutZip_path"]),pathLab=str(cfg["camera"]["ia_labelTxt_path"]),cam_index=int(cfg["camera"]["right_index"]), safeZone=float(cfg["security"]["border_sec"]),frame_rate=int(cfg["camera"]["frame_rate"]))  # droite

# =============================
# Flux vidéo Flask
# =============================
@app.route("/video0")
def video0():
    return Response(
        camG.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/video1")
def video1():
    return Response(
        camD.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/")
def index():
    return render_template_string("""
    <html>
        <body>
            <h1>Flux caméras</h1>
            <img src="/video0">
            <img src="/video1">
        </body>
    </html>
    """)

# =============================
# Pairing gauche/droite
# =============================
def find_best_pair(dets_left, dets_right, max_dist_px=PAIR_MAX_DIST_PX):
    """
    dets_left / dets_right : liste de dicts renvoyés par get_detections()
      {
        "box": (x1, y1, x2, y2),
        "score": float,
        "class_id": int,
        "label": str,
        "center": (cx, cy),
      }

    Retourne (det_left, det_right) de distance minimale, ou None.
    """
    best_pair = None
    best_d2 = max_dist_px
    for dl in dets_left:
        cx_l, cy_l = dl["center"]
        for dr in dets_right:
            cx_r, cy_r = dr["center"]
            dx = cx_l - cx_r
            dy = cy_l - cy_r
            d2 = 0.8*(dy * dy) + 0.2*(dx * dx) # ? si meme ligne force a nous
            if dy < best_d2:
                best_d2 = d2
                best_pair = (dl, dr)

    return best_pair

# =============================
# Boucle de tir (thread)
# =============================
def shooting_loop():
    """
    - lit les détections IA sur cam gauche/droite
    - fait les paires sur les centres les plus proches
    - calcule pitch/yaw
    - oriente le galvo, allume le laser, "shoot", attend 1s
    """

    while True:
        # 1) détection gauche
        dets_left = camG.get_detections(score_threshold=SCORE_MIN_DETECTION)
        if not dets_left:
            time.sleep(0.05)
            continue

        # 2) détection droite
        dets_right = camD.get_detections(score_threshold=SCORE_MIN_DETECTION)
        if not dets_right:
            time.sleep(0.05)
            continue

        # 3) meilleure paire sur centre le plus proche
        pair = find_best_pair(dets_left, dets_right)
        if pair is None:
            # rien de cohérent entre gauche et droite
            time.sleep(0.05)
            continue

        det_l, det_r = pair
        lx, ly = det_l["center"]
        rx, ry = det_r["center"]

        # 4) calcul angles + distance
        try:
            pitch_deg, yaw_deg, distance_m = calc.compute_angles(
                lx=lx, ly=ly, rx=rx, ry=ry
            )
        except Exception as e:
            print(f"[WARN] échec compute_angles: {e}")
            time.sleep(0.05)
            continue

        # 5) pilotage galvo
        # Convention : theta_x = yaw (horizontal), theta_y = pitch (vertical)
        galvo.set_angles(theta_x=yaw_deg, theta_y=pitch_deg)
        time.sleep(0.05)
        # 6) tir
        galvo.laser_on()
        print("shoot")
        time.sleep(TEMPS_LASER_SHOOT)
        galvo.laser_off()
        # temps attempste 2 fois laser
        time.sleep(TEMPS_LASER_SHOOT+TEMPS_LASER_SHOOT)

        # petite pause pour ne pas saturer
        # time.sleep(0.05)

# =============================
# gestion signal CTRL+C
# =============================
def signal_handler(sig, frame):
    print("[INFO] Arrêt")
    camG.stop()
    camD.stop()
    galvo.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


# =============================
# lancement
# =============================
if __name__ == "__main__":
    #! importer le fichier config 
    
    # thread de tir
    shoot_thread = threading.Thread(target=shooting_loop, daemon=True)
    shoot_thread.start()

    # serveur Flask (stream vidéo)
    app.run(host="0.0.0.0", port=5000)

