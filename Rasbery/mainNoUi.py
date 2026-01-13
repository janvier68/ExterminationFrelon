import signal
import sys
import time
import threading

from Galvo import GalvoController
from Cam import CameraManager
from Profondeur import StereoAngleCalculator
from Config import load_config, gpio_from_string

# =============================
# paramètres
# =============================
cfg = load_config("config/config.json")

SCORE_MIN_DETECTION = float(cfg["detection"]["score_min_detection"])
PAIR_MAX_DIST_PX = int(cfg["detection"]["pair_max_dist_px"])
TEMPS_LASER_SHOOT = float(cfg["laser"]["shoot_time_s"])

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
camG = CameraManager(
    cam_index=int(cfg["camera"]["left_index"]),
    safeZone=float(cfg["security"]["border_sec"]),
    frame_rate=int(cfg["camera"]["frame_rate"]),
)
camD = CameraManager(
    cam_index=int(cfg["camera"]["right_index"]),
    safeZone=float(cfg["security"]["border_sec"]),
    frame_rate=int(cfg["camera"]["frame_rate"]),
)

# =============================
# Pairing gauche/droite (optimisé)
# =============================
def find_best_pair(dets_left, dets_right, max_dist_px=PAIR_MAX_DIST_PX):
    """
    Retourne (det_left, det_right) minimisant un coût basé sur l'écart vertical (prioritaire)
    et horizontal (secondaire), avec gate sur la distance verticale.
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
# Boucle principale (sans Flask, sans GUI)
# =============================
stop_event = threading.Event()

def shooting_loop():
    """
    Boucle unique pour minimiser overhead:
    - récupère détections gauche/droite
    - paire la plus cohérente
    - calcule angles
    - pilote galvo + tir
    """
    sleep_idle = 0.01  # réduit CPU quand rien à faire
    settle_time = 0.02 # temps de stabilisation galvo (réduit)

    while not stop_event.is_set():
        dets_left = camG.get_detections(score_threshold=SCORE_MIN_DETECTION)
        if not dets_left:
            time.sleep(sleep_idle)
            continue

        dets_right = camD.get_detections(score_threshold=SCORE_MIN_DETECTION)
        if not dets_right:
            time.sleep(sleep_idle)
            continue

        pair = find_best_pair(dets_left, dets_right)
        if pair is None:
            time.sleep(sleep_idle)
            continue

        det_l, det_r = pair
        lx, ly = det_l["center"]
        rx, ry = det_r["center"]

        try:
            pitch_deg, yaw_deg, distance_m = calc.compute_angles(lx=lx, ly=ly, rx=rx, ry=ry)
        except Exception:
            time.sleep(sleep_idle)
            continue

        galvo.set_angles(theta_x=yaw_deg, theta_y=pitch_deg)
        time.sleep(settle_time)

        galvo.laser_on()
        time.sleep(TEMPS_LASER_SHOOT)
        galvo.laser_off()

        # Cooldown anti double tir (garde ton comportement original)
        cooldown = TEMPS_LASER_SHOOT * 2.0
        end_t = time.monotonic() + cooldown
        while not stop_event.is_set() and time.monotonic() < end_t:
            time.sleep(0.01)

def shutdown():
    stop_event.set()
    try:
        camG.stop()
    except Exception:
        pass
    try:
        camD.stop()
    except Exception:
        pass
    try:
        galvo.shutdown()
    except Exception:
        pass

def signal_handler(sig, frame):
    shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    # thread unique (daemon) + blocage du main thread
    t = threading.Thread(target=shooting_loop, daemon=True)
    t.start()

    # main thread idle (faible overhead)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        shutdown()
