
import numpy as np


class StereoAngleCalculator:
    """
    Classe pour calculer la 3D + les angles (pitch/yaw) d’un point
    à partir de ses coordonnées pixel dans les deux caméras (gauche/droite).
    """

    def __init__(
        self,
        baseline_m: float = 0.10,
        h_cl_m: float = 0.0, # hauteur entre cam laser
        focal_length_mm: float = 4.74,
        pixel_size_mm: float = 0.00155,
    ):
        # Constantes optiques (fallback datasheet)
        self.focal_length_mm = focal_length_mm
        self.pixel_size_mm = pixel_size_mm
        self.focal_length_pixels = self.focal_length_mm / self.pixel_size_mm

        # Baseline (sera écrasée par la calibration si T dispo)
        self.baseline_m = baseline_m

        # Décalage vertical laser/caméra (laser en dessous caméra)
        self.h_cl_m = h_cl_m

        # Paramètres intrinsèques rectifiés (caméra gauche)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Cartes de rectification (optionnel)
        self.map1x = None
        self.map1y = None
        self.map2x = None
        self.map2y = None

        # Taille image (utile pour fallback si pas de calibration)
        self.image_width = None
        self.image_height = None

    # ------------------------------------------------------------------ #
    #  Chargement calibration stéréo
    # ------------------------------------------------------------------ #
    def load_stereo_calibration(self, path: str) -> bool:
        """
        Charge les fichiers de calibration stéréo (.npz) et met à jour:
        - maps de rectification
        - fx, fy, cx, cy (caméra gauche rectifiée)
        - baseline_m (norme de T)
        """
        try:
            data = np.load(path)

            # Maps de rectification
            self.map1x = data["map1x"]
            self.map1y = data["map1y"]
            self.map2x = data["map2x"]
            self.map2y = data["map2y"]

            # Matrice de projection P1 (caméra gauche rectifiée)
            P1 = data["P1"]
            self.fx = float(P1[0, 0])
            self.fy = float(P1[1, 1])
            self.cx = float(P1[0, 2])
            self.cy = float(P1[1, 2])

            # Translation T (vecteur de base stéréo)
            T = data["T"]  # en mètres si la calibration a été faite en mètres
            self.baseline_m = float(np.linalg.norm(T.reshape(-1)))

            # Déduction taille image à partir des maps si dispo
            if self.map1x is not None:
                self.image_height, self.image_width = self.map1x.shape[:2]

            print(
                f"[INFO] Calibration stéréo chargée: fx={self.fx:.1f}, "
                f"fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}, "
                f"baseline={self.baseline_m*100:.1f} cm"
            )
            return True

        except Exception as e:
            print(f"[WARN] Impossible de charger '{path}': {e}")
            return False

    # ------------------------------------------------------------------ #
    #  Définir la taille image (utile en fallback)
    # ------------------------------------------------------------------ #
    def set_image_size(self, width: int, height: int):
        """
        Définit la taille de l'image pour pouvoir utiliser le fallback datasheet
        si aucune calibration n'a été chargée (fx,fy,cx,cy None).
        """
        self.image_width = width
        self.image_height = height

        # Si aucun fx/fy/cx/cy n'a été défini, on en déduit un fallback
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            f_px = self.focal_length_pixels
            self.fx = f_px
            self.fy = f_px
            self.cx = width / 2.0
            self.cy = height / 2.0

            print(
                f"[INFO] Fallback intrinsèques: fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}"
            )

    # ------------------------------------------------------------------ #
    #  Calcul des angles + distance
    # ------------------------------------------------------------------ #
    def compute_angles(
        self, lx: float, ly: float, rx: float, ry: float, debug:bool=1
    ):
        """
        Calcule:
          - la position 3D du point (X,Y,Z) dans le repère caméra gauche
          - l'angle de pitch et de yaw du laser pour viser ce point
          - la distance du point à la caméra (norme 3D)

        Entrées:
          lx, ly : coordonnées pixel dans l'image gauche
          rx, ry : coordonnées pixel dans l'image droite
        Sortie:
          pitch_deg, yaw_deg, distance_m
        """

        # Vérif des intrinsèques
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            raise RuntimeError(
                "Intrinsèques fx,fy,cx,cy non définis. "
                "Charge une calibration ou appelle set_image_size()."
            )

        # Disparité (attention au signe)
        disparity = lx - rx
        if disparity < 0.5: # ? il y avait abs avant 
            raise ValueError(f"Disparité trop faible: {disparity:.3f} px")

        # Triangulation simple (caméras rectifiées, baseline horizontale)
        Z = (self.fx * self.baseline_m) / disparity  # profondeur (m)
        X = (lx - self.cx) * Z / self.fx
        Y = (ly - self.cy) * Z / self.fy

        # Distance 3D
        Pr = float(np.sqrt(X * X + Y * Y + Z * Z))

        # Sécurité
        if abs(Pr) < 1e-9:
            raise ValueError("Distance 3D trop faible (Pr ≈ 0).")

        # Opposés pour calcul d'angle (on travaille en mètres)
        opp_vert_m = float(Y)  # vertical
        opp_horiz_m = float(X)  # horizontal

        # Clamp pour arcsin
        ratio_vert = float(np.clip(opp_vert_m / Pr, -1.0, 1.0))
        ratio_horiz = float(np.clip(opp_horiz_m / Pr, -1.0, 1.0))

        # Vue de côté: composante projetée
        add = float(np.cos(np.arcsin(ratio_vert)) * Pr)

        # Pitch = atan2( (H_CL + Y), Add )
        laser_pitch_rad = float(np.arctan2(self.h_cl_m + opp_vert_m, add))

        # Vue de haut: yaw = asin( X / Pr )
        laser_yaw_rad = float(np.arcsin(ratio_horiz))

        deg = 180.0 / np.pi
        laser_pitch_deg = laser_pitch_rad * deg
        laser_yaw_deg = laser_yaw_rad * deg

        # Debug console
        if debug :
            print(
                f"[3D] X={X:.3f} m, Y={Y:.3f} m, Z={Z:.3f} m, "
                f"disp={disparity:.2f} px, dist={Pr:.3f} m"
            )
            print(f"[ANGLE] pitch={laser_pitch_deg:.2f}°, yaw={laser_yaw_deg:.2f}°")

        return laser_pitch_deg, laser_yaw_deg, Pr


##### utilisation
if __name__ == "__main__":

    calc = StereoAngleCalculator(baseline_m=0.10, h_cl_m=0.0)

    # 1) Charger la calibration (si dispo)
    calc.load_stereo_calibration("calibation_photos/visualisation_calibration/stereo_params.npz")

    # 2) Si pas de calibration, définir au moins la taille d’image pour fallback
    # calc.set_image_size(width=640, height=480)

    # 3) Calculer les angles à partir des coordonnées pixel (gauche/droite)
    pitch_deg, yaw_deg, distance_m = calc.compute_angles(lx, ly, rx, ry)
