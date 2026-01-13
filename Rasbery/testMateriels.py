from flask import Flask, Response, render_template_string
import signal
import sys
import time
import threading

from Galvo import GalvoController
from Cam import CameraManager
from Profondeur import StereoAngleCalculator
from Config import load_config, gpio_from_string

app = Flask(__name__)

# =============================
# paramètres
# =============================
cfg = load_config("config/config.json")

TEMPS_LASER_SHOOT = float(cfg["laser"]["shoot_time_s"])
DISPLAY_HZ = float(cfg["camera"].get("frame_rate", 30))
DISPLAY_PERIOD_S = 1.0 / max(1.0, DISPLAY_HZ)

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
# Profondeur + angles (chargée comme dans ton code)
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
safe_zone = float(cfg["security"]["border_sec"])
frame_rate = int(cfg["camera"]["frame_rate"])

camG = CameraManager(
    cam_index=int(cfg["camera"]["left_index"]),
    safeZone=safe_zone,
    frame_rate=frame_rate
)
camD = CameraManager(
    cam_index=int(cfg["camera"]["right_index"]),
    safeZone=safe_zone,
    frame_rate=frame_rate
)

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
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                .row { display: flex; gap: 16px; }
                .col { display: flex; flex-direction: column; }
                img { border: 1px solid #333; max-width: 48vw; height: auto; }
                .msg { margin-top: 12px; padding: 10px; background: #111; color: #0f0; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Test composant (centre caméra / laser au centre)</h1>
            <div class="row">
                <div class="col">
                    <h2>Cam gauche</h2>
                    <img src="/video0">
                </div>
                <div class="col">
                    <h2>Cam droite</h2>
                    <img src="/video1">
                </div>
            </div>
            <div class="msg">
                Le point au centre de l'image (cam gauche) et le laser doivent être identiques.
            </div>
        </body>
    </html>
    """)

# =============================
# Utilitaires centre image
# =============================
def center_of_image(cam: CameraManager):
    """
    Récupère la taille d'image depuis l'objet CameraManager.
    Essaie plusieurs noms d'attributs courants pour rester robuste.
    Retour: (cx, cy)
    """
    w = None
    h = None

    # cas fréquents
    for a_w, a_h in [
        ("width", "height"),
        ("W", "H"),
        ("frame_width", "frame_height"),
        ("img_w", "img_h"),
    ]:
        if hasattr(cam, a_w) and hasattr(cam, a_h):
            w = int(getattr(cam, a_w))
            h = int(getattr(cam, a_h))
            break

    # si CameraManager expose une dernière frame (selon ton implémentation)
    if (w is None or h is None) and hasattr(cam, "last_frame") and cam.last_frame is not None:
        try:
            h = int(cam.last_frame.shape[0])
            w = int(cam.last_frame.shape[1])
        except Exception:
            pass

    if w is None or h is None:
        # fallback (à ajuster si besoin)
        # évite de crasher, mais mieux vaut que CameraManager expose width/height
        w, h = 640, 480

    return (w // 2, h // 2)

# =============================
# Boucle de test (thread)
# =============================
def center_test_loop():
    """
    - oriente le galvo à (0,0) (centre)
    - allume le laser (optionnel: pulsé)
    - affiche en continu le centre de l'image de la cam gauche
    - message: le centre cam et le laser doivent coïncider
    """
    # 1) centre (angles 0,0)
    galvo.set_angles(theta_x=0.0, theta_y=0.0)
    time.sleep(0.1)

    # 2) laser: soit ON continu, soit pulsé (choisir)
    LASER_CONTINU = False

    if LASER_CONTINU:
        galvo.laser_on()
        print("[TEST] Laser ON continu au centre (theta_x=0, theta_y=0)")
    else:
        print("[TEST] Laser pulsé au centre (theta_x=0, theta_y=0)")

    try:
        while True:
            cx, cy = center_of_image(camG)

            # affichage console
            print(
                f"[TEST] Centre image camG = ({cx},{cy}) | "
                f"Laser au centre -> doit être identique"
            )

            # si pulsé
            if not LASER_CONTINU:
                galvo.laser_on()
                time.sleep(TEMPS_LASER_SHOOT)
                galvo.laser_off()

            time.sleep(DISPLAY_PERIOD_S)
    finally:
        # sécurité si sortie non prévue
        try:
            galvo.laser_off()
        except Exception:
            pass

# =============================
# gestion signal CTRL+C
# =============================
def signal_handler(sig, frame):
    print("[INFO] Arrêt")
    try:
        camG.stop()
    except Exception:
        pass
    try:
        camD.stop()
    except Exception:
        pass
    try:
        galvo.laser_off()
    except Exception:
        pass
    try:
        galvo.shutdown()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# =============================
# lancement
# =============================
if __name__ == "__main__":
    test_thread = threading.Thread(target=center_test_loop, daemon=True)
    test_thread.start()

    app.run(host="0.0.0.0", port=5000)
