import cv2
import torch
import numpy as np
import requests
import io

from ultralytics.models.sam import SAM3SemanticPredictor


# ============================================================
# CONFIG
# ============================================================

SAM_MODEL = "sam3.pt"

# Your FastAPI Depth Pro server
DEPTH_API_URL = "http://localhost:8000/predict-depth-map"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class SpoonDepthEstimator:

    def __init__(
        self,
        sam_model_path,
        depth_api_url=DEPTH_API_URL,
        device=None
    ):
        self.sam_model_path = sam_model_path
        self.depth_api_url = depth_api_url

        self.device = (
            device
            if device
            else torch.device(
                "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        )

        # Load SAM only once
        self.sam_predictor = self.load_sam_predictor()

    # ========================================================
    # LOAD SAM
    # ========================================================

    def load_sam_predictor(self):

        print("\nLoading SAM 3...")

        overrides = {
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "model": self.sam_model_path,
            "quantize": 16,
        }

        return SAM3SemanticPredictor(
            overrides=overrides
        )

    # ========================================================
    # GET DEPTH MAP FROM FASTAPI
    # ========================================================

    def get_depth_from_api(self, image_path):

        print("\nSending image to Depth Pro API...")

        with open(image_path, "rb") as f:

            files = {
                "file": (
                    image_path,
                    f,
                    "image/jpeg"
                )
            }

            response = requests.post(
                self.depth_api_url,
                files=files,
                timeout=600
            )

        if response.status_code != 200:

            raise RuntimeError(
                f"Depth API failed "
                f"({response.status_code}): "
                f"{response.text}"
            )

        # Response contains numpy depth map
        depth_map = np.load(
            io.BytesIO(response.content)
        )

        print(
            f"Depth map received: "
            f"{depth_map.shape}"
        )

        return depth_map

    # ========================================================
    # ESTIMATE DEPTH
    # ========================================================

    def estimate_depth(self, image_path):

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image_height, image_width = image_rgb.shape[:2]

        # ----------------------------------------------------
        # Detect spoon using SAM 3
        # ----------------------------------------------------

        print("\nRunning SAM 3...")

        self.sam_predictor.set_image(image_path)

        results = self.sam_predictor(
            text=["spoon"]
        )

        if not results:
            raise RuntimeError(
                "SAM 3 returned no results."
            )

        result = results[0]

        if result.masks is None:
            raise RuntimeError(
                "SAM 3 did not generate any spoon masks."
            )

        masks = result.masks.data

        masks_np = (
            masks.detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Find largest spoon mask
        # ----------------------------------------------------

        mask_areas = [
            np.sum(mask > 0.5)
            for mask in masks_np
        ]

        largest_idx = np.argmax(mask_areas)

        spoon_mask = (
            masks_np[largest_idx] > 0.5
        )

        # ----------------------------------------------------
        # Resize mask if necessary
        # ----------------------------------------------------

        if spoon_mask.shape != (
            image_height,
            image_width
        ):

            spoon_mask = cv2.resize(
                spoon_mask.astype(np.uint8),
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        # ----------------------------------------------------
        # Get Depth Pro depth map from FastAPI
        # ----------------------------------------------------

        depth_map = self.get_depth_from_api(
            image_path
        )

        # ----------------------------------------------------
        # Make sure depth map matches image
        # ----------------------------------------------------

        if depth_map.shape != (
            image_height,
            image_width
        ):

            print(
                f"Resizing depth map "
                f"{depth_map.shape} -> "
                f"{(image_height, image_width)}"
            )

            depth_map = cv2.resize(
                depth_map,
                (image_width, image_height),
                interpolation=cv2.INTER_CUBIC
            )

        # ----------------------------------------------------
        # Apply spoon mask
        # ----------------------------------------------------

        spoon_depth_values = depth_map[
            spoon_mask
        ]

        if len(spoon_depth_values) == 0:

            raise RuntimeError(
                "Spoon mask contains no pixels."
            )

        # ----------------------------------------------------
        # Depth statistics
        # ----------------------------------------------------

        spoon_min_depth = (
            spoon_depth_values.min()
        )

        spoon_max_depth = (
            spoon_depth_values.max()
        )

        raw_depth_difference = (
            spoon_max_depth -
            spoon_min_depth
        )

        return {

            "depth_map": depth_map,

            "spoon_mask": spoon_mask,

            "spoon_min_depth": spoon_min_depth,

            "spoon_max_depth": spoon_max_depth,

            "raw_depth_difference":
                raw_depth_difference
        }

    # ========================================================
    # FIND FARTHEST SPOON PIXELS
    # ========================================================

    def find_farthest_spoon_pixels(
        self,
        spoon_mask,
        depth_map
    ):

        spoon_coords = np.column_stack(
            np.where(spoon_mask)
        )

        if len(spoon_coords) < 2:

            raise RuntimeError(
                "Not enough spoon pixels "
                "to calculate distance."
            )

        candidate_points = []

        # ----------------------------------------------------
        # Find extreme points in multiple directions
        # ----------------------------------------------------

        for angle in np.linspace(
            0,
            2 * np.pi,
            32,
            endpoint=False
        ):

            direction = np.array([
                np.cos(angle),
                np.sin(angle)
            ])

            projections = (
                spoon_coords @ direction
            )

            candidate_points.append(
                spoon_coords[
                    np.argmin(projections)
                ]
            )

            candidate_points.append(
                spoon_coords[
                    np.argmax(projections)
                ]
            )

        candidate_points = np.unique(
            np.array(candidate_points),
            axis=0
        )

        # ----------------------------------------------------
        # Find maximum pixel distance
        # ----------------------------------------------------

        max_distance = -1

        point1 = None
        point2 = None

        for i in range(
            len(candidate_points)
        ):

            for j in range(
                i + 1,
                len(candidate_points)
            ):

                y1, x1 = candidate_points[i]
                y2, x2 = candidate_points[j]

                distance = np.sqrt(
                    (x2 - x1) ** 2 +
                    (y2 - y1) ** 2
                )

                if distance > max_distance:

                    max_distance = distance

                    point1 = (
                        int(y1),
                        int(x1)
                    )

                    point2 = (
                        int(y2),
                        int(x2)
                    )

        # ----------------------------------------------------
        # Get Depth Pro depth at those points
        # ----------------------------------------------------

        depth1 = float(
            depth_map[point1]
        )

        depth2 = float(
            depth_map[point2]
        )

        depth_difference = abs(
            depth1 - depth2
        )

        return {

            "point1": point1,

            "point2": point2,

            "max_distance":
                float(max_distance),

            "depth1":
                depth1,

            "depth2":
                depth2,

            "depth_difference":
                float(depth_difference)
        }

    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================

    def process_image(self, image_path):

        results = self.estimate_depth(
            image_path
        )

        depth_map = results[
            "depth_map"
        ]

        spoon_mask = results[
            "spoon_mask"
        ]

        farthest_pixels_info = (
            self.find_farthest_spoon_pixels(
                spoon_mask,
                depth_map
            )
        )

        return {

            "results": results,

            "farthest_pixels_info":
                farthest_pixels_info
        }
