import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.models.sam import SAM3SemanticPredictor


class FoodSegmenter:

    def __init__(
        self,
        yolo_model_path,
        sam_model_path="sam3.pt",
        conf=0.5,
        quantize=16
    ):

        self.yolo_model_path = yolo_model_path
        self.sam_model_path = sam_model_path
        self.conf = conf
        self.quantize = quantize

        # ----------------------------------------------------
        # Load models
        # ----------------------------------------------------

        self.yolo_model = self._load_yolo()
        self.sam_predictor = self._load_sam()

    # ========================================================
    # LOAD YOLO
    # ========================================================

    def _load_yolo(self):

        print("\nLoading YOLO model...")

        model = YOLO(
            self.yolo_model_path
        )

        print("YOLO model loaded successfully")

        return model

    # ========================================================
    # LOAD SAM 3
    # ========================================================

    def _load_sam(self):

        print("\nLoading SAM 3...")

        overrides = {
            "conf": self.conf,
            "task": "segment",
            "mode": "predict",
            "model": self.sam_model_path,
            "quantize": self.quantize,
        }

        predictor = SAM3SemanticPredictor(
            overrides=overrides
        )

        print("SAM 3 loaded successfully")

        return predictor

    # ========================================================
    # SEGMENT FOOD + CALCULATE GEOMETRY
    # ========================================================

    def segment_food(self, image_path):

        print("\n======================================")
        print("PROCESSING FOOD IMAGE")
        print("======================================")

        # ----------------------------------------------------
        # Run SAM 3
        # ----------------------------------------------------

        self.sam_predictor.set_image(
            image_path
        )

        results = self.sam_predictor(
            text=["food"]
        )

        if not results:
            raise RuntimeError(
                "SAM 3 returned no results."
            )

        result = results[0]

        # ----------------------------------------------------
        # Check masks
        # ----------------------------------------------------

        if result.masks is None:
            raise RuntimeError(
                "SAM 3 did not generate any food masks."
            )

        masks = result.masks.data

        print(
            f"Number of masks detected: "
            f"{len(masks)}"
        )

        # ----------------------------------------------------
        # Convert masks to NumPy
        # ----------------------------------------------------

        masks_np = (
            masks
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Calculate area of every mask
        # ----------------------------------------------------

        mask_areas = [
            np.sum(mask > 0.5)
            for mask in masks_np
        ]

        # ----------------------------------------------------
        # Select largest food mask
        # ----------------------------------------------------

        largest_idx = np.argmax(
            mask_areas
        )

        food_mask = (
            masks_np[largest_idx] > 0.5
        )

        # ----------------------------------------------------
        # Convert mask to uint8
        # ----------------------------------------------------

        food_mask_uint8 = (
            food_mask.astype(np.uint8)
            * 255
        )

        # ====================================================
        # FIND FOOD CONTOUR
        # ====================================================

        contours, _ = cv2.findContours(
            food_mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            raise RuntimeError(
                "Could not find food contour."
            )

        # ----------------------------------------------------
        # Select largest contour
        # ----------------------------------------------------

        food_contour = max(
            contours,
            key=cv2.contourArea
        )

        # ====================================================
        # CALCULATE FOOD GEOMETRY
        # ====================================================

        circumference_pixels = (
            cv2.arcLength(
                food_contour,
                closed=True
            )
        )

        area_pixels = (
            cv2.contourArea(
                food_contour
            )
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        print("\n======================================")
        print("FOOD GEOMETRY")
        print("======================================")

        print(
            f"Food area         : "
            f"{area_pixels:.2f} pixels²"
        )

        print(
            f"Food circumference: "
            f"{circumference_pixels:.2f} pixels"
        )

        return {
            "result": result,

            "masks": masks,

            "masks_np": masks_np,

            "food_mask": food_mask,

            "food_mask_uint8":
                food_mask_uint8,

            "food_contour":
                food_contour,

            "area_pixels":
                float(area_pixels),

            "circumference_pixels":
                float(circumference_pixels),

            "largest_mask_index":
                int(largest_idx)
        }