# stereo_calib.py
import cv2
import numpy as np
import glob
import os

# Paramètres du damier (intérieur, nb coins)
PATTERN_SIZE = (9, 6)      # 10 coins horizontal, 7 vertical -> adapter
SQUARE_SIZE  = 0.025        # 2,5 cm entre centres de cases -> adapter

# Dossiers d'images
CALIBFILE = "photos_calibration"
LEFT_GLOB  = f"{CALIBFILE}/left_*.png"
RIGHT_GLOB = f"{CALIBFILE}/right_*.png"

# Dossier de visu (images côte à côte avec damier dessiné)
DEBUG_DIR = os.path.join(CALIBFILE, "visualisation_calibration")
os.makedirs(DEBUG_DIR, exist_ok=True)

# Points 3D du damier dans son repère (z=0)
objp = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []   # points 3D (par image)
imgpointsL = []  # points 2D caméra gauche
imgpointsR = []  # points 2D caméra droite

left_files  = sorted(glob.glob(LEFT_GLOB))
right_files = sorted(glob.glob(RIGHT_GLOB))
assert len(left_files) == len(right_files) and len(left_files) > 0, "Nb d'images gauche/droite incohérent ou nul"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

for idx, (lf, rf) in enumerate(zip(left_files, right_files)):
    imgL = cv2.imread(lf, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
    assert imgL is not None and imgR is not None, f"Erreur lecture {lf}/{rf}"

    retL, cornersL = cv2.findChessboardCorners(imgL, PATTERN_SIZE)
    retR, cornersR = cv2.findChessboardCorners(imgR, PATTERN_SIZE)
    if not (retL and retR):
        print(f"[WARN] damier non trouvé pour {lf}/{rf}")
        continue

    # Affinage sub-pixel
    cornersL = cv2.cornerSubPix(imgL, cornersL, (11, 11), (-1, -1), criteria)
    cornersR = cv2.cornerSubPix(imgR, cornersR, (11, 11), (-1, -1), criteria)

    objpoints.append(objp)
    imgpointsL.append(cornersL)
    imgpointsR.append(cornersR)
    print(f"[OK] damier trouvé sur {lf} / {rf}")

    # ---------- VISU : image gauche/droite côte à côte avec damier dessiné ----------
    # passer les images en BGR pour dessiner en couleur
    visL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
    visR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

    cv2.drawChessboardCorners(visL, PATTERN_SIZE, cornersL, True)
    cv2.drawChessboardCorners(visR, PATTERN_SIZE, cornersR, True)

    # concat horizontale
    concat = np.hstack((visL, visR))

    out_name = os.path.join(DEBUG_DIR, f"pair_{idx:02d}.png")
    cv2.imwrite(out_name, concat)
    print(f"[VISU] sauvegardé {out_name}")

# Si aucun damier valide trouvé
assert len(objpoints) > 0, "Aucun damier valide trouvé sur les paires d'images"

h, w = imgL.shape[:2]
image_size = (w, h)

# 1) Calibration mono de chaque caméra
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, image_size, None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, image_size, None, None)

print("[INFO] Calibration mono OK")
print("mtxL:\n", mtxL)
print("mtxR:\n", mtxR)

# 2) Calibration stéréo
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
flags = cv2.CALIB_FIX_INTRINSIC  # on garde mtxL,mtxR fixes

retS, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    image_size,
    criteria=criteria_stereo,
    flags=flags
)

print("[INFO] Calibration stéréo OK, RMS =", retS)
print("R:\n", R)
print("T:\n", T)

# 3) Rectification + maps
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtxL, distL, mtxR, distR,
    image_size, R, T, alpha=0
)

map1x, map1y = cv2.initUndistortRectifyMap(
    mtxL, distL, R1, P1, image_size, cv2.CV_32FC1
)
map2x, map2y = cv2.initUndistortRectifyMap(
    mtxR, distR, R2, P2, image_size, cv2.CV_32FC1
)

np.savez("stereo_params.npz",
         mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR,
         R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
         map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
         image_size=image_size)

print("[INFO] Paramètres stéréo enregistrés dans stereo_params.npz")
os.replace("stereo_params.npz", os.path.join(DEBUG_DIR, "stereo_params.npz"))
print(f"[INFO] Images de visu enregistrées dans {DEBUG_DIR}")