import numpy as np
import os
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_yolo_ultralytics

class YOLO(Model):
    def __init__(self):
        super().__init__(
            model_file=os.path.expanduser("~/Exterminateur/ia/best_imx_model/packerOut.zip"),
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )

        self.labels = np.genfromtxt(
            os.path.expanduser("~/Exterminateur/ia/best_imx_model/labels.txt"),
            dtype=str,
            delimiter="\n",
        )

    def post_process(self, output_tensors):
        return pp_od_yolo_ultralytics(output_tensors)

    @staticmethod
    def get_box_center(box):
        """
        box = (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) * 0.5)
        cy = int((y1 + y2) * 0.5)
        return cx, cy
