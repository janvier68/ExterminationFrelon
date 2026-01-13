import threading
import cv2

from modlib.devices import AiCamera
from modlib.apps import Annotator
from .yolo_model import YOLO


class CameraManager:
    def __init__(self,pathZip,pathLab, cam_index=0, frame_rate=8, on_detection=None,safeZone=0.20,):
        self.device = AiCamera(num=cam_index, frame_rate=frame_rate)
        self.model = YOLO(pathZip,pathLab) # charger le modèle ia
        self.annotator = Annotator()
        self.lock = threading.Lock()

        self.safeZone = safeZone
        self.recSafeZone = None
        # callback
        self.on_detection = on_detection

        self.device.deploy(self.model)
        self.device.start()

        self.aim = None


    def stop(self):
        self.device.stop()

    def _get_safe_zone_rect(self, w=0, h=0):
        """
        Retourne le rectangle de safe zone en pixels.
        safeZone = fraction de padding sur chaque bord
        """
        if self.recSafeZone == None:
            pad_x = int(self.safeZone * w)
            pad_y = int(self.safeZone * h)

            x1 = pad_x
            y1 = pad_y
            x2 = w - pad_x
            y2 = h - pad_y
            
            self.recSafeZone = (x1, y1, x2, y2)

        return self.recSafeZone


    def generate_frames(self):
        """
        Générateur MJPEG
        """

        while True:
            with self.lock:
                frame_obj = next(self.device)
                frame = frame_obj.image

            if frame_obj.detections and len(frame_obj.detections) > 0:
                labels = []
                centers = []

                for box, score, class_id, _ in frame_obj.detections:
                    labels.append(
                        f"{self.model.labels[class_id]}: {score:.2f}"
                    )

                    centers.append(
                        self.model.get_box_center(tuple(box))
                    )

                self.annotator.annotate_boxes(
                    frame_obj,
                    frame_obj.detections,
                    labels=labels,
                    alpha=0.3,
                    corner_radius=10
                )

                frame = frame_obj.image

                if self.on_detection:
                    self.on_detection(centers)

            # ---- DESSIN DE LA SAFE ZONE (rectangle central) ----
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = self._get_safe_zone_rect(w, h)
            # rectangle vert (épaisseur 2 px)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    def get_detections(self, score_threshold: float = 0.5):

        with self.lock:
            frame_obj = next(self.device)

        # Récupération de la taille image
        # adapte selon ton objet : frame / image / rgb / np_array
        if hasattr(frame_obj, "frame"):
            img = frame_obj.frame
        else:
            img = frame_obj.image

        h, w = img.shape[:2]

        detections = []

        for box, score, class_id, _ in frame_obj.detections:
            if score < score_threshold:
                continue

            # dénormalisation si les coords sont en [0,1]
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
            
            # centre du bounding box en pixels
            cx, cy = self.model.get_box_center((x1, y1, x2, y2))
            
            # saf carré
            x_safe1, y_safe1, x_safe2, y_safe2 = self._get_safe_zone_rect(w, h)

            # Si le centre est DANS le padding (en dehors du rectangle central), on ignore
            if cx < x_safe1 or cx > x_safe2 or cy < y_safe1 or cy > y_safe2:
                continue

            detections.append(
                {
                    "score": float(score),
                    "class_id": int(class_id),
                    "label": self.model.labels[class_id],
                    "center": (cx, cy),
                }
            )

        return detections
