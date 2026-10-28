
from ultralytics import YOLO
import cv2
import numpy as np
import math


# ============================================================
# PATHS
# ============================================================

SPOON_MODEL_PATH = (
    r"C:\Study\Major\runs\detect\runs\spoon_yolo11m\weights\last.pt"
)

FOOD_MODEL_PATH = (
    r"C:\Study\Major\Food\classif.pt"
)

IMAGE_PATH = r"Spoon/Tests/Test3.jpeg"

print("Loading models...")

def load_model(model_path):
    model = YOLO(model_path)
    return model
class SpoonDetection:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def detect_spoon(self, image_path):
        results = self.model(image_path)
        best_spoon = None
        best_confidence = 0.0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                confidence = float(box.conf[0])

                if confidence > best_confidence:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    best_spoon = {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width_px": x2 - x1,
                        "height_px": y2 - y1,
                        "confidence": confidence
                    }
                    best_confidence = confidence

        if best_spoon is None:
            raise RuntimeError("Spoon was not detected in the image.")

        spoon_width_px = best_spoon["width_px"]
        spoon_height_px = best_spoon["height_px"]
        spoon_length_px = max(spoon_width_px, spoon_height_px)
        result = {
                "spoon_length_px": spoon_length_px,
                "spoon_height_px": spoon_height_px,
                "spoon_width_px": spoon_width_px,
                "spoon_confidence": best_confidence
            }
        return result
   