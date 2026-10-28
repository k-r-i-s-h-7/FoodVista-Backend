# ============================================================
# FOOD AREA + DENSITY + NUTRITION ESTIMATION PIPELINE
# Box-guided SAM3 segmentation
# ============================================================

import cv2
import json
import numpy as np
import requests
import io

from pathlib import Path
from ultralytics import YOLO
from Food_detection import FoodDetection
from Spoon_detection import SpoonDetection
from FoodAreap import FoodSegmenter
from SpoonLength import SpoonDepthEstimator


class FoodAreaPipeline:
    """
    Pipeline:
      1. Detect food items with YOLO.
      2. Detect a spoon/fork for scale calibration.
      3. Segment the utensil with its detection box.
      4. Retrieve the Depth Pro map from FastAPI.
      5. Create a depth-adjusted cm/pixel scale map.
      6. Segment each food using its own YOLO bounding box.
      7. Estimate food area and approximate volume.
      8. Estimate mass from configured density ranges.
      9. Estimate nutrients from the mixture database.
     10. Estimate visible ingredient-area percentages.

    Important:
      - UTENSIL_LENGTH_CM is an assumed reference length.
      - CONE_HEIGHT_CM is a fixed approximation.
      - A single image does not directly reveal true food volume.
      - Ingredient area percentages are visual estimates, not
        ingredient mass percentages.
    """

    UTENSIL_LENGTH_CM = 15.0
    CONE_HEIGHT_CM = 5.0

    def __init__(
        self,
        food_model_path,
        utensil_model_path,
        sam_model_path=r"C:\Study\Major\sam3.pt",
        depth_api_url="http://localhost:8000/predict-depth-map",
        food_confidence=0.50,
        utensil_confidence=0.25,
        ingredient_model_path=None,
        density_data_path=(
            r"C:\Study\Major\Working\Pipeline\Metrics\Density.json"
        ),
        density_classes_path=(
            r"C:\Study\Major\Working\Pipeline\Metrics\DensityRange.json"
        ),
        nutrition_db_path=(
            r"C:\Study\Major\Working\Pipeline\Metrics\Ingredients.json"
        ),
        single_items_path=(
            r"C:\Study\Major\Working\Pipeline\Metrics\SingleItems.json"
        ),
    ):
        print("\nInitializing food estimation pipeline...")

        # ---------------- JSON CONFIGURATION ----------------
        self.nutrition_db = self._load_json(
            nutrition_db_path, "nutrition database"
        )
        self.ingredient_profiles = self.nutrition_db.get(
            "_ingredient_profiles_per_100g", {}
        )
        self.nutrients = self.nutrition_db.get(
            "_metadata", {}
        ).get("nutrients", [])

        if not self.ingredient_profiles or not self.nutrients:
            raise ValueError(
                "Ingredients.json must contain "
                "'_ingredient_profiles_per_100g' and "
                "'_metadata.nutrients'."
            )

        self.density_data = self._load_json(
            density_data_path, "food density mapping"
        )
        self.density_classes = self._load_json(
            density_classes_path, "density class ranges"
        )
        self.single_item_weights = self._load_json(
            single_items_path, "single-item weights"
        )

        # ---------------- FOOD DETECTOR ----------------
        self.food_detector = FoodDetection(food_model_path)

        # ---------------- UTENSIL DETECTOR ----------------
        self.utensil_detector = SpoonDetection(utensil_model_path)
        self.utensil_confidence = utensil_confidence

        # ---------------- SAM3 PREDICTOR ----------------
        # Uses the SAM3SemanticPredictor already initialized
        # by your FoodSegmenter class in FoodAreap.py.
        self.food_segmenter = FoodSegmenter(
            yolo_model_path=food_model_path,
            sam_model_path=sam_model_path,
            conf=food_confidence,
        )

        # ---------------- DEPTH PRO CLIENT ----------------
        self.depth_estimator = SpoonDepthEstimator(
            sam_model_path=sam_model_path,
            depth_api_url=depth_api_url,
        )

        # ---------------- INGREDIENT DETECTOR ----------------
        if ingredient_model_path is None:
            raise ValueError(
                "ingredient_model_path is required."
            )

        self.ingredient_model = YOLO(ingredient_model_path)

        print("Pipeline initialized successfully.")

    # ========================================================
    # JSON + CONFIGURATION HELPERS
    # ========================================================

    @staticmethod
    def _load_json(path, description):
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Could not find {description}: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _normalize_food_name(name):
        return (
            str(name)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

    def _dish_record(self, food_type):
        key = self._normalize_food_name(food_type)
        record = self.nutrition_db.get(key)

        if not isinstance(record, dict):
            raise KeyError(
                f"Food '{food_type}' is missing from Ingredients.json "
                f"(normalized key: '{key}')."
            )

        if "base_type" not in record or "ingredients" not in record:
            raise KeyError(
                f"Food record '{key}' must contain 'base_type' "
                "and 'ingredients'."
            )

        return key, record

    def _get_density_class(self, food_type):
        key = self._normalize_food_name(food_type)

        record = self.density_data.get(key)
        if record is None:
            record = self.density_data.get(
                str(food_type).strip().lower()
            )

        if not isinstance(record, dict):
            raise KeyError(
                f"No density mapping found for '{food_type}'."
            )

        density_class = record.get("density_class")

        if not density_class:
            raise KeyError(
                f"Density mapping for '{food_type}' has no "
                "'density_class'."
            )

        return density_class

    def _get_density_bounds(self, density_class):
        record = self.density_classes.get(density_class)

        if not isinstance(record, dict):
            raise KeyError(
                f"Density class '{density_class}' is missing "
                "from DensityRange.json."
            )

        try:
            lower = float(record["lower_bound"])
            upper = float(record["upper_bound"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Density class '{density_class}' must contain "
                "numeric lower_bound and upper_bound."
            ) from exc

        if lower < 0 or upper < lower:
            raise ValueError(
                f"Invalid density range for '{density_class}': "
                f"{lower}, {upper}."
            )

        return lower, upper

    def _get_single_item_weight(self, food_type):
        key = self._normalize_food_name(food_type)

        value = self.single_item_weights.get(key)
        if value is None:
            value = self.single_item_weights.get(
                str(food_type).strip().lower()
            )

        if isinstance(value, dict):
            value = value.get(
                "weight_g",
                value.get("base_weight_g", value.get("weight")),
            )

        if value is None:
            raise KeyError(
                f"No single-item weight for '{food_type}' "
                "was found in SingleItems.json."
            )

        weight = float(value)

        if weight <= 0:
            raise ValueError(
                f"Single-item weight for '{food_type}' "
                "must be positive."
            )

        return weight

    # ========================================================
    # NUTRITION ESTIMATION
    # ========================================================

    def calculate_dish_nutrients(
        self,
        dish_name,
        target_weight_g=100.0,
    ):
        """
        Calculate nutrients using configured ingredient fractions.

        Ingredient nutrient profiles must be per 100 g.
        For base_type='other', use the configured single-item weight.
        """

        if target_weight_g <= 0:
            raise ValueError(
                "target_weight_g must be greater than zero."
            )

        key, dish = self._dish_record(dish_name)
        recipe = dish["ingredients"]
        base_type = str(dish["base_type"]).strip().lower()

        if not recipe:
            raise ValueError(
                f"No ingredients defined for '{key}'."
            )

        if base_type == "other":
            actual_weight_g = self._get_single_item_weight(key)
            weight_source = "assumed_single_item_weight"
        else:
            actual_weight_g = float(target_weight_g)
            weight_source = "volume_density_estimate"

        proportions = {}

        for ingredient, fraction in recipe.items():
            if ingredient not in self.ingredient_profiles:
                raise KeyError(
                    f"Missing nutrient profile for '{ingredient}'."
                )

            if not isinstance(fraction, (int, float)) or fraction < 0:
                raise ValueError(
                    f"Invalid recipe proportion for '{ingredient}': "
                    f"{fraction}"
                )

            proportions[ingredient] = float(fraction)

        total_fraction = sum(proportions.values())

        if total_fraction <= 0:
            raise ValueError(
                f"Ingredient proportions for '{key}' must sum "
                "to a positive value."
            )

        # Normalize proportions in case they do not sum exactly to 1.
        proportions = {
            name: value / total_fraction
            for name, value in proportions.items()
        }

        total_nutrients = {
            nutrient: 0.0 for nutrient in self.nutrients
        }
        ingredient_breakdown = {}

        for ingredient, fraction in proportions.items():
            quantity_g = fraction * actual_weight_g
            profile = self.ingredient_profiles[ingredient]
            contribution = {}

            for nutrient in self.nutrients:
                value_per_100g = float(
                    profile.get(nutrient, 0.0)
                )
                amount = value_per_100g * quantity_g / 100.0

                contribution[nutrient] = amount
                total_nutrients[nutrient] += amount

            ingredient_breakdown[ingredient] = {
                "quantity_g": round(quantity_g, 4),
                "nutrients": {
                    name: round(value, 4)
                    for name, value in contribution.items()
                },
            }

        return {
            "dish": key,
            "base_type": base_type,
            "weight_source": weight_source,
            "actual_weight_g": round(actual_weight_g, 4),
            "ingredients": ingredient_breakdown,
            "total_nutrients": {
                name: round(value, 4)
                for name, value in total_nutrients.items()
            },
            "nutrients_per_100g": {
                name: round(
                    value * 100.0 / actual_weight_g, 4
                )
                for name, value in total_nutrients.items()
            },
        }

    # ========================================================
    # MASK UTILITIES
    # ========================================================

    @staticmethod
    def _resize_mask(mask, image_shape):
        """Return a boolean mask matching the original image size."""
        height, width = image_shape[:2]
        mask = np.asarray(mask).squeeze()

        if mask.ndim != 2:
            raise ValueError(
                f"Expected a 2D mask; received shape {mask.shape}."
            )

        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        return mask.astype(bool)

    @staticmethod
    def _clip_bbox(bbox, image_shape):
        """Clip an XYXY box to the original image dimensions."""
        height, width = image_shape[:2]
        x1, y1, x2, y2 = map(float, bbox)

        x1 = max(0, min(int(round(x1)), width - 1))
        y1 = max(0, min(int(round(y1)), height - 1))
        x2 = max(x1 + 1, min(int(round(x2)), width))
        y2 = max(y1 + 1, min(int(round(y2)), height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid bounding box after clipping: "
                f"{(x1, y1, x2, y2)}"
            )

        return x1, y1, x2, y2

    @classmethod
    def _select_mask_for_box(
        cls,
        masks,
        bbox,
        image_shape,
    ):
        """
        Select the candidate mask most consistent with one detection.

        Score combines:
          - box coverage: fraction of the detection box covered
          - mask purity: fraction of the mask inside the detection box

        The mask must already be in original-image coordinates.
        """

        x1, y1, x2, y2 = cls._clip_bbox(
            bbox, image_shape
        )
        height, width = image_shape[:2]
        box_area = max((x2 - x1) * (y2 - y1), 1)

        best_mask = None
        best_score = -1.0
        best_metrics = None

        for raw_mask in masks:
            mask = cls._resize_mask(raw_mask, image_shape)
            mask_area = int(mask.sum())

            if mask_area == 0:
                continue

            intersection = int(
                mask[y1:y2, x1:x2].sum()
            )

            if intersection == 0:
                continue

            box_coverage = intersection / box_area
            mask_purity = intersection / mask_area

            # Favor box coverage but penalize masks extending
            # far beyond the current food's detection box.
            score = (
                0.60 * box_coverage
                + 0.40 * mask_purity
            )

            if score > best_score:
                best_score = score
                best_mask = mask
                best_metrics = {
                    "box_coverage": float(box_coverage),
                    "mask_purity": float(mask_purity),
                    "score": float(score),
                }

        if best_mask is None:
            raise RuntimeError(
                "SAM3 returned no non-empty mask overlapping "
                f"the bounding box {(x1, y1, x2, y2)}."
            )

        return best_mask, best_metrics

    # ========================================================
    # STAGE 1: FOOD DETECTION
    # ========================================================

    def detect_food(self, image_path):
        print("\n[STAGE 1] Detecting food items...")

        foods = self.food_detector.detect_food(image_path)

        for index, food in enumerate(foods, start=1):
            required = (
                "name", "confidence", "x1", "y1", "x2", "y2"
            )

            missing = [
                key for key in required if key not in food
            ]

            if missing:
                raise KeyError(
                    f"Food detection {index} is missing keys: "
                    f"{missing}"
                )

            print(
                f"  {index}. {food['name']} "
                f"(confidence={float(food['confidence']):.3f})"
            )

        return foods

    # ========================================================
    # STAGE 2: UTENSIL DETECTION
    # ========================================================

    def detect_utensil(self, image_path):
        print("\n[STAGE 2] Detecting spoon/fork...")

        model = self.utensil_detector.model
        results = model.predict(
            source=image_path,
            conf=self.utensil_confidence,
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                name = str(model.names[class_id]).strip().lower()

                if "spoon" not in name and "fork" not in name:
                    continue

                x1, y1, x2, y2 = map(
                    float, box.xyxy[0].tolist()
                )

                detections.append({
                    "name": name,
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                })

        if not detections:
            raise RuntimeError(
                "No spoon/fork detected. Check the utensil model's "
                "class names and confidence threshold."
            )

        # The existing scale-calibration method uses one utensil.
        utensil = max(
            detections,
            key=lambda item: item["confidence"],
        )

        print(
            f"  Using {utensil['name']} "
            f"(confidence={utensil['confidence']:.3f})"
        )

        return utensil

    # ========================================================
    # STAGE 3: BOX-GUIDED UTENSIL SEGMENTATION
    # ========================================================

    def segment_utensil(self, image_path, utensil):
        print("\n[STAGE 3] Segmenting utensil using its box...")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        predictor = self.food_segmenter.sam_predictor
        predictor.set_image(image_path)

        # SAM3SemanticPredictor accepts XYXY pixel boxes.
        # labels=[1] marks the box as a positive visual prompt.
        results = predictor(
            bboxes=[list(utensil["bbox"])],
            labels=[1],
        )

        if not results or results[0].masks is None:
            raise RuntimeError(
                f"SAM3 returned no mask for {utensil['name']}."
            )

        raw_masks = (
            results[0].masks.data.detach().cpu().numpy()
        )

        mask, metrics = self._select_mask_for_box(
            masks=raw_masks,
            bbox=utensil["bbox"],
            image_shape=image.shape,
        )

        if int(mask.sum()) < 2:
            raise RuntimeError(
                "Utensil mask contains too few pixels."
            )

        # PCA estimates the major-axis length of the utensil mask.
        ys, xs = np.where(mask)
        points = np.column_stack((xs, ys)).astype(np.float32)

        center = points.mean(axis=0)
        centered = points - center

        _, _, vh = np.linalg.svd(
            centered, full_matrices=False
        )

        principal_axis = vh[0]
        projections = centered @ principal_axis

        point1 = points[np.argmin(projections)]
        point2 = points[np.argmax(projections)]

        length_px = float(
            np.linalg.norm(point2 - point1)
        )

        if length_px <= 0:
            raise RuntimeError(
                "Could not calculate utensil pixel length."
            )

        print(f"  Utensil mask pixels: {int(mask.sum())}")
        print(f"  Utensil length: {length_px:.2f} px")
        print(f"  Mask match score: {metrics['score']:.3f}")

        return {
            "mask": mask,
            "length_px": length_px,
            "endpoint1_xy": tuple(map(float, point1)),
            "endpoint2_xy": tuple(map(float, point2)),
            "mask_match_metrics": metrics,
        }

    # ========================================================
    # STAGE 4: DEPTH PRO MAP
    # ========================================================

    def get_depth_map(self, image_path, image_shape):
        print("\n[STAGE 4] Requesting Depth Pro map...")

        depth_map = self.depth_estimator.get_depth_from_api(
            image_path
        )

        if depth_map is None:
            raise RuntimeError(
                "Depth Pro API returned no depth map."
            )

        depth_map = np.asarray(depth_map)

        if depth_map.ndim != 2:
            raise RuntimeError(
                f"Expected a 2D depth map; got {depth_map.shape}."
            )

        height, width = image_shape[:2]

        if depth_map.shape != (height, width):
            depth_map = cv2.resize(
                depth_map.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        depth_map = depth_map.astype(np.float32)
        valid = np.isfinite(depth_map) & (depth_map > 0)

        if not valid.any():
            raise RuntimeError(
                "Depth Pro returned no valid positive depth values."
            )

        print(f"  Depth map shape: {depth_map.shape}")
        print(
            f"  Valid depth range: "
            f"{depth_map[valid].min():.4f} to "
            f"{depth_map[valid].max():.4f}"
        )

        return depth_map

    # ========================================================
    # STAGE 5: DEPTH-ADJUSTED SCALE MAP
    # ========================================================

    def calculate_scale_map(
        self,
        depth_map,
        utensil_mask,
        utensil_length_px,
    ):
        print("\n[STAGE 5] Calculating depth-adjusted scale...")

        valid = np.isfinite(depth_map) & (depth_map > 0)
        utensil_depths = depth_map[utensil_mask & valid]

        if utensil_depths.size == 0:
            raise RuntimeError(
                "No valid depth values inside the utensil mask."
            )

        reference_depth = float(np.median(utensil_depths))

        reference_cm_per_pixel = (
            self.UTENSIL_LENGTH_CM / utensil_length_px
        )

        scale_map = np.full(
            depth_map.shape, np.nan, dtype=np.float32
        )

        scale_map[valid] = (
            reference_cm_per_pixel
            * depth_map[valid]
            / reference_depth
        )

        print(
            f"  Assumed utensil length: "
            f"{self.UTENSIL_LENGTH_CM:.2f} cm"
        )
        print(f"  Reference depth: {reference_depth:.4f}")
        print(
            f"  Reference scale: "
            f"{reference_cm_per_pixel:.6f} cm/pixel"
        )

        return {
            "scale_map": scale_map,
            "reference_depth": reference_depth,
            "reference_cm_per_pixel": reference_cm_per_pixel,
        }

    # ========================================================
    # STAGE 6: BOX-GUIDED SEGMENTATION FOR EACH FOOD
    # ========================================================

    def segment_food_item(
        self,
        image_path,
        food,
        image_shape,
    ):
        """
        Segment one YOLO food detection using its own box prompt.

        Inputs:
            image_path: original image path
            food: dictionary returned by FoodDetection.detect_food()
                  containing name, confidence, x1, y1, x2, y2
            image_shape: original image.shape

        Returns:
            Boolean mask in original-image coordinates.
        """

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        height, width = image_shape[:2]

        bbox = self._clip_bbox(
            (
                food["x1"],
                food["y1"],
                food["x2"],
                food["y2"],
            ),
            image_shape,
        )

        x1, y1, x2, y2 = bbox
        predictor = self.food_segmenter.sam_predictor

        # Important: set the full original image, not a crop.
        # Therefore the YOLO box coordinates remain valid.
        predictor.set_image(image_path)

        # Box-guided SAM3 segmentation. Do NOT use text=["food"]
        # here: the box is the prompt for this specific detection.
        results = predictor(
            bboxes=[[float(x1), float(y1), float(x2), float(y2)]],
            labels=[1],
        )

        if not results or results[0].masks is None:
            raise RuntimeError(
                f"SAM3 returned no box-guided mask for "
                f"'{food['name']}' at {bbox}."
            )

        raw_masks = (
            results[0].masks.data.detach().cpu().numpy()
        )

        # Match candidate masks to this detection's box.
        # This is important when SAM3 returns more than one mask.
        mask, metrics = self._select_mask_for_box(
            masks=raw_masks,
            bbox=bbox,
            image_shape=image_shape,
        )

        if not mask.any():
            raise RuntimeError(
                f"Selected mask for '{food['name']}' is empty."
            )

        print(
            f"  Segmented '{food['name']}': "
            f"{int(mask.sum())} pixels; "
            f"box coverage={metrics['box_coverage']:.3f}; "
            f"mask purity={metrics['mask_purity']:.3f}; "
            f"score={metrics['score']:.3f}"
        )

        # Return only the mask, so downstream stages receive
        # the same return type as the existing pipeline.
        return mask

    # ========================================================
    # STAGE 7: FOOD AREA
    # ========================================================

    def calculate_food_area(self, food_mask, scale_map):
        valid_pixels = (
            food_mask
            & np.isfinite(scale_map)
            & (scale_map > 0)
        )

        pixel_count = int(valid_pixels.sum())

        if pixel_count == 0:
            raise RuntimeError(
                "No food-mask pixels have valid depth."
            )

        pixel_area_map = np.square(scale_map)

        area_cm2 = float(
            np.sum(pixel_area_map[valid_pixels])
        )

        return {
            "area_pixels": pixel_count,
            "area_cm2": area_cm2,
            "food_mask": food_mask,
        }

    # ========================================================
    # STAGE 8: FOOD VOLUME
    # ========================================================

    def calculate_food_volume(self, base_area_cm2):
        """
        Cone approximation:
            V = (1/3) * base area * assumed height
        """
        return float(
            base_area_cm2 * self.CONE_HEIGHT_CM / 3.0
        )

    # ========================================================
    # STAGE 9: TOTAL FOOD MASK
    # ========================================================

    def build_total_food_mask(self, food_results, image_shape):
        height, width = image_shape[:2]
        total_food_mask = np.zeros(
            (height, width), dtype=bool
        )

        for food in food_results:
            if "food_mask" not in food:
                continue

            mask = self._resize_mask(
                food["food_mask"], image_shape
            )
            total_food_mask |= mask

        if not total_food_mask.any():
            raise RuntimeError(
                "No valid food masks are available for "
                "ingredient analysis."
            )

        return total_food_mask

    # ========================================================
    # STAGE 10: INGREDIENT DETECTION
    # ========================================================

    def detect_ingredients(
        self,
        image_path,
        total_food_mask,
        image_shape,
        confidence_threshold=0.25,
    ):
        height, width = image_shape[:2]

        results = self.ingredient_model.predict(
            source=image_path,
            conf=confidence_threshold,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        model_masks = None

        if result.masks is not None:
            model_masks = (
                result.masks.data.detach().cpu().numpy()
            )

        names = self.ingredient_model.names
        detections = []

        for index, box in enumerate(result.boxes):
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            ingredient_name = str(names[class_id])

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            if model_masks is not None and index < len(model_masks):
                ingredient_mask = self._resize_mask(
                    model_masks[index] > 0.5,
                    image_shape,
                )
            else:
                # Detection-only checkpoint fallback.
                # A box is less precise than a segmentation mask.
                ingredient_mask = np.zeros(
                    (height, width), dtype=bool
                )

                x1 = max(0, min(x1, width))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height))
                y2 = max(0, min(y2, height))

                if x2 > x1 and y2 > y1:
                    ingredient_mask[y1:y2, x1:x2] = True

            # Exclude pixels outside all segmented food regions.
            ingredient_mask &= total_food_mask

            if not ingredient_mask.any():
                continue

            detections.append({
                "ingredient": ingredient_name,
                "class_id": class_id,
                "confidence": confidence,
                "mask": ingredient_mask,
            })

        return detections

    # ========================================================
    # STAGE 11: INGREDIENT AREA PERCENTAGES
    # ========================================================

    def calculate_ingredient_percentages(
        self,
        detections,
        total_food_mask,
        scale_map=None,
    ):
        valid_food = total_food_mask.copy()

        if scale_map is not None:
            valid_food &= (
                np.isfinite(scale_map) & (scale_map > 0)
            )
            pixel_weights = np.square(scale_map)
        else:
            pixel_weights = np.ones(
                total_food_mask.shape, dtype=np.float32
            )

        total_area = float(
            pixel_weights[valid_food].sum()
        )

        if total_area <= 0:
            raise RuntimeError(
                "Total food area is zero."
            )

        # Assign an overlapping pixel to the highest-confidence
        # ingredient detection, preventing double counting.
        confidence_map = np.full(
            total_food_mask.shape,
            -1.0,
            dtype=np.float32,
        )
        owner_map = np.full(
            total_food_mask.shape,
            -1,
            dtype=np.int32,
        )

        for index, detection in enumerate(detections):
            mask = detection["mask"] & valid_food
            update = mask & (
                detection["confidence"] > confidence_map
            )

            confidence_map[update] = detection["confidence"]
            owner_map[update] = index

        summary = {}

        for index, detection in enumerate(detections):
            assigned = owner_map == index

            if not assigned.any():
                continue

            name = detection["ingredient"]
            area = float(pixel_weights[assigned].sum())
            pixels = int(assigned.sum())

            if name not in summary:
                summary[name] = {
                    "ingredient": name,
                    "area_pixels": 0,
                    "area_cm2": 0.0,
                    "detection_count": 0,
                    "confidence_sum": 0.0,
                }

            summary[name]["area_pixels"] += pixels
            summary[name]["area_cm2"] += area
            summary[name]["detection_count"] += 1
            summary[name]["confidence_sum"] += (
                detection["confidence"]
            )

        ingredient_results = []

        for item in summary.values():
            item["area_percentage"] = (
                item["area_cm2"] / total_area * 100.0
            )
            item["mean_confidence"] = (
                item["confidence_sum"]
                / item["detection_count"]
            )
            del item["confidence_sum"]
            ingredient_results.append(item)

        ingredient_results.sort(
            key=lambda item: item["area_percentage"],
            reverse=True,
        )

        assigned_area = sum(
            item["area_cm2"] for item in ingredient_results
        )

        return {
            "total_food_area": total_area,
            "ingredients": ingredient_results,
            "unassigned_area_percentage": max(
                0.0,
                (total_area - assigned_area) / total_area * 100.0,
            ),
        }

    def analyze_ingredient_percentages(
        self,
        image_path,
        food_results,
        image_shape,
        scale_map=None,
    ):
        print("\n[STAGE 10] Building total food mask...")

        total_food_mask = self.build_total_food_mask(
            food_results, image_shape
        )

        print("\n[STAGE 11] Detecting ingredient regions...")

        detections = self.detect_ingredients(
            image_path=image_path,
            total_food_mask=total_food_mask,
            image_shape=image_shape,
        )

        if not detections:
            print("  No ingredient detections found.")
            return {
                "total_food_mask": total_food_mask,
                "ingredient_detections": [],
                "ingredient_summary": [],
                "unassigned_area_percentage": 100.0,
            }

        summary = self.calculate_ingredient_percentages(
            detections=detections,
            total_food_mask=total_food_mask,
            scale_map=scale_map,
        )

        for item in summary["ingredients"]:
            print(
                f"  {item['ingredient']}: "
                f"{item['area_percentage']:.2f}%"
            )

        print(
            "  Unassigned area: "
            f"{summary['unassigned_area_percentage']:.2f}%"
        )

        return {
            "total_food_mask": total_food_mask,
            "ingredient_detections": detections,
            "ingredient_summary": summary["ingredients"],
            "total_food_area": summary["total_food_area"],
            "unassigned_area_percentage": (
                summary["unassigned_area_percentage"]
            ),
        }

    # ========================================================
    # MAIN PIPELINE
    # ========================================================
    def calculate_blended_nutrients(
        self,
        dish_name,
        target_weight_g,
        ingredient_summary,
        unassigned_area_percentage,
    ):
        """
        Estimate nutrients using detected ingredient area percentages.

        Detected ingredients use their individual nutrient profiles.
        The remaining area uses the dish's configured recipe profile.

        Area percentages are approximations of mass percentages.
        """

        key, dish = self._dish_record(dish_name)

        if target_weight_g <= 0:
            raise ValueError("target_weight_g must be positive.")

        base_type = str(dish["base_type"]).strip().lower()

        # Use the existing recipe as the fallback profile.
        # This avoids requiring a separate profile for every gravy/dal.
        base_nutrients = self.calculate_dish_nutrients(
            key,
            target_weight_g=100.0,
        )["nutrients_per_100g"]

        contributions = {}
        used_percentages = {}
        total_nutrients_per_100g = {
            nutrient: 0.0 for nutrient in self.nutrients
        }

        detected_percentage = 0.0
        fallback_percentage = float(unassigned_area_percentage)

        for item in ingredient_summary:
            ingredient = self._normalize_food_name(
                item["ingredient"]
            )
            percentage = float(item["area_percentage"])

            if percentage <= 0:
                continue

            profile = self.ingredient_profiles.get(ingredient)

            # If no profile exists, use the fallback dish profile
            # for that portion instead of losing its nutrients.
            if not isinstance(profile, dict):
                fallback_percentage += percentage
                continue

            detected_percentage += percentage
            used_percentages[ingredient] = (
                used_percentages.get(ingredient, 0.0) + percentage
            )

        # Detected ingredients: weighted nutrient contribution.
        for ingredient, percentage in used_percentages.items():
            profile = self.ingredient_profiles[ingredient]
            fraction = percentage / 100.0

            ingredient_nutrients = {}

            for nutrient in self.nutrients:
                value = float(profile.get(nutrient, 0.0))
                amount = fraction * value

                total_nutrients_per_100g[nutrient] += amount
                ingredient_nutrients[nutrient] = round(amount, 4)

            contributions[ingredient] = {
                "area_percentage": round(percentage, 4),
                "nutrients_per_100g_mixture": ingredient_nutrients,
            }

        # Undetected region: use the dish's recipe-derived profile.
        fallback_fraction = max(0.0, fallback_percentage) / 100.0

        for nutrient in self.nutrients:
            total_nutrients_per_100g[nutrient] += (
                fallback_fraction * float(
                    base_nutrients.get(nutrient, 0.0)
                )
            )

        total_nutrients = {
            nutrient: value * target_weight_g / 100.0
            for nutrient, value in total_nutrients_per_100g.items()
        }

        return {
            "dish": key,
            "base_type": base_type,
            "actual_weight_g": round(target_weight_g, 4),
            "method": "ingredient_area_weighted",
            "detected_ingredient_percentage": round(
                detected_percentage, 4
            ),
            "fallback_percentage": round(
                fallback_percentage, 4
            ),
            "ingredient_contributions": contributions,
            "total_nutrients": {
                name: round(value, 4)
                for name, value in total_nutrients.items()
            },
            "nutrients_per_100g": {
                name: round(value, 4)
                for name, value in total_nutrients_per_100g.items()
            },
        }
    def estimate(self, image_path, model=False):
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        image_shape = image.shape

        # ---------------- FOOD DETECTION ----------------
        detected_foods = self.detect_food(image_path)

        if not detected_foods:
            raise RuntimeError(
                "No food items were detected."
            )

        food_results = []
        geometry_foods = []

        # Preserve the detector order in the final result.
        for food in detected_foods:
            food_type = food["name"]
            key, dish_record = self._dish_record(food_type)

            item = {
                "food_type": food_type,
                "confidence": float(food["confidence"]),
                "bbox": (
                    float(food["x1"]),
                    float(food["y1"]),
                    float(food["x2"]),
                    float(food["y2"]),
                ),
                "base_type": str(
                    dish_record["base_type"]
                ).strip().lower(),
            }

            if item["base_type"] == "other":
                # Count-type/single-item foods use SingleItems.json.
                item["geometry_skipped"] = True
                item["geometry_skip_reason"] = (
                    "base_type is 'other'; using SingleItems.json weight"
                )
                item["estimated_weight_g"] = (
                    self._get_single_item_weight(key)
                )
                item["nutrient_estimate"] = (
                    self.calculate_dish_nutrients(
                        key,
                        target_weight_g=item["estimated_weight_g"],
                    )
                )

                print(
                    f"  {food_type}: assumed weight "
                    f"{item['estimated_weight_g']:.1f} g"
                )
            else:
                item["geometry_skipped"] = False
                geometry_foods.append((food, item))

            food_results.append(item)

        # If all detected items are single-item foods, no geometry
        # calibration is required.
        if not geometry_foods:
            return {
                "foods": food_results,
                "geometry_skipped_for_all_foods": True,
                "ingredient_analysis": None,
                "ingredient_summary": [],
            }

        # ---------------- UTENSIL + DEPTH CALIBRATION ----------------
        utensil = self.detect_utensil(image_path)
        utensil_data = self.segment_utensil(
            image_path, utensil
        )
        depth_map = self.get_depth_map(
            image_path, image_shape
        )

        calibration = self.calculate_scale_map(
            depth_map=depth_map,
            utensil_mask=utensil_data["mask"],
            utensil_length_px=utensil_data["length_px"],
        )

        scale_map = calibration["scale_map"]

        # ---------------- FOOD SEGMENTATION + MASS ----------------
        print(
            "\n[STAGE 6] Box-guided food segmentation, "
            "area, volume, density and nutrition..."
        )

        measured_food_results = []

        for food, item in geometry_foods:
            try:
                food_mask = self.segment_food_item(
                    image_path=image_path,
                    food=food,
                    image_shape=image_shape,
                )

                measurements = self.calculate_food_area(
                    food_mask=food_mask,
                    scale_map=scale_map,
                )

                area_cm2 = measurements["area_cm2"]
                volume_cm3 = self.calculate_food_volume(
                    area_cm2
                )

                item.update({
                    "food_mask": measurements["food_mask"],
                    "food_area_pixels": measurements["area_pixels"],
                    "food_area_cm2": area_cm2,
                    "volume_cm3": volume_cm3,
                })

                # Estimate mass from density range midpoint.
                density_class = self._get_density_class(
                    item["food_type"]
                )
                lower_density, upper_density = (
                    self._get_density_bounds(density_class)
                )
                mean_density = (
                    lower_density + upper_density
                ) / 2.0

                estimated_weight_g = (
                    volume_cm3 * mean_density
                )

                item.update({
                    "density_class": density_class,
                    "density_range_g_cm3": [
                        lower_density, upper_density
                    ],
                    "density_used_g_cm3": mean_density,
                    "estimated_weight_g": round(
                        estimated_weight_g, 4
                    ),
                    "nutrient_estimate": (
                        self.calculate_dish_nutrients(
                            item["food_type"],
                            target_weight_g=estimated_weight_g,
                        )
                    ),
                })

                measured_food_results.append(item)

                print(
                    f"  {item['food_type']}: "
                    f"{area_cm2:.2f} cm², "
                    f"{volume_cm3:.2f} cm³, "
                    f"~{estimated_weight_g:.1f} g"
                )

            except (RuntimeError, KeyError, ValueError) as exc:
                print(
                    f"  Could not fully estimate "
                    f"{item['food_type']}: {exc}"
                )
                item["estimation_error"] = str(exc)

        # ---------------- INGREDIENT ANALYSIS ----------------
        ingredient_analysis = None
        ingredient_summary = []

        if model and measured_food_results:
            print("\n[STAGE 10] Ingredient-aware nutrition estimation...")

            # Analyze each food separately so that dal, rice, and gravy
            # use their own dish-specific fallback recipe.
            for item in measured_food_results:
                analysis = self.analyze_ingredient_percentages(
                    image_path=image_path,
                    food_results=[item],
                    image_shape=image_shape,
                    scale_map=scale_map,
                )

                item["ingredient_analysis"] = {
                    "ingredient_summary": analysis["ingredient_summary"],
                    "unassigned_area_percentage": (
                        analysis["unassigned_area_percentage"]
                    ),
                }

                item["nutrient_estimate"] = (
                    self.calculate_blended_nutrients(
                        dish_name=item["food_type"],
                        target_weight_g=item["estimated_weight_g"],
                        ingredient_summary=analysis["ingredient_summary"],
                        unassigned_area_percentage=(
                            analysis["unassigned_area_percentage"]
                        ),
                    )
                )

                ingredient_summary.append({
                    "food_type": item["food_type"],
                    "ingredients": analysis["ingredient_summary"],
                    "unassigned_area_percentage": (
                        analysis["unassigned_area_percentage"]
                    ),
                })

            ingredient_analysis = {
                "per_food": ingredient_summary,
            }

        output = {
            "foods": food_results,
            "geometry_skipped_for_all_foods": False,
            "utensil": utensil,
            "utensil_mask": utensil_data["mask"],
            "utensil_length_px": utensil_data["length_px"],
            "assumed_utensil_length_cm": self.UTENSIL_LENGTH_CM,
            "depth_map": depth_map,
            "scale_map_cm_per_pixel": scale_map,
            "reference_depth": calibration["reference_depth"],
            "reference_cm_per_pixel": (
                calibration["reference_cm_per_pixel"]
            ),
            "ingredient_analysis": ingredient_analysis,
            "ingredient_summary": ingredient_summary,
        }

        if ingredient_analysis is not None:
            output["total_food_mask"] = None

        return output


# # ============================================================
# # RUN CONFIGURATION
# # ============================================================

# FOOD_MODEL_PATH = r"C:\Study\Major\Working\Pipeline\models\classif.pt"

# UTENSIL_MODEL_PATH = (
#     r"C:\Study\Major\Working\Pipeline\models\spoon_model.pt"
# )

# SAM_MODEL_PATH = r"C:\Study\Major\Working\Pipeline\models\sam3.pt"

# IMAGE_PATH = r"C:\Study\Major\Spoon\Tests\Test2.jpeg"

# INGREDIENT_MODEL_PATH = r"C:\Study\Major\Working\Pipeline\models\ing_model.pt"

# DEPTH_API_URL = "http://localhost:8000/predict-depth-map"

# DENSITY_DATA_PATH = (
#     r"C:\Study\Major\Working\Pipeline\Metrics\Density.json"
# )

# DENSITY_CLASSES_PATH = (
#     r"C:\Study\Major\Working\Pipeline\Metrics\DensityRange.json"
# )

# NUTRITION_DB_PATH = (
#     r"C:\Study\Major\Working\Pipeline\Metrics\Ingredients.json"
# )

# SINGLE_ITEMS_PATH = (
#     r"C:\Study\Major\Working\Pipeline\Metrics\SingleItems.json"
# )


# # ============================================================
# # INITIALIZE AND RUN
# # ============================================================

# pipeline = FoodAreaPipeline(
#     food_model_path=FOOD_MODEL_PATH,
#     utensil_model_path=UTENSIL_MODEL_PATH,
#     sam_model_path=SAM_MODEL_PATH,
#     depth_api_url=DEPTH_API_URL,
#     food_confidence=0.50,
#     utensil_confidence=0.25,
#     ingredient_model_path=INGREDIENT_MODEL_PATH,
#     density_data_path=DENSITY_DATA_PATH,
#     density_classes_path=DENSITY_CLASSES_PATH,
#     nutrition_db_path=NUTRITION_DB_PATH,
#     single_items_path=SINGLE_ITEMS_PATH,
# )

# output = pipeline.estimate(IMAGE_PATH)


# # ============================================================
# # PRINT FINAL RESULTS
# # ============================================================

# print("\n========== FINAL RESULTS ==========")

# for item in output["foods"]:
#     print(
#         f"\nFood: {item['food_type']} "
#         f"(base_type={item['base_type']})"
#     )

#     if item.get("geometry_skipped"):
#         print(
#             f"  Geometry: skipped "
#             f"({item['geometry_skip_reason']})"
#         )

#     elif "food_area_cm2" in item:
#         print(
#             f"  Area: {item['food_area_cm2']:.2f} cm²"
#         )
#         print(
#             f"  Volume: {item['volume_cm3']:.2f} cm³"
#         )

#         if "density_used_g_cm3" in item:
#             print(
#                 f"  Density: "
#                 f"{item['density_used_g_cm3']:.4f} g/cm³"
#             )

#     if "estimated_weight_g" in item:
#         print(
#             f"  Estimated serving weight: "
#             f"{item['estimated_weight_g']:.2f} g"
#         )

#     if "nutrient_estimate" in item:
#         print("  Total nutrients:")

#         for nutrient, amount in (
#             item["nutrient_estimate"]["total_nutrients"].items()
#         ):
#             print(f"    {nutrient}: {amount}")

#     if "estimation_error" in item:
#         print(
#             f"  Estimation warning: {item['estimation_error']}"
#         )

# if output.get("ingredient_summary"):
#     print("\n========== INGREDIENT AREA BREAKDOWN ==========")

#     for item in output["ingredient_summary"]:
#         print(
#             f"{item['ingredient']}: "
#             f"{item['area_percentage']:.2f}%"
#         )

#     if output.get("ingredient_analysis"):
#         print(
#             "Unassigned area: "
#             f"{output['ingredient_analysis']}"
#             "['unassigned_area_percentage']:.2f}%"
#         )
# else:
#     print(
#         "\nIngredient area analysis skipped "
#         "(no geometry-measured foods or no ingredient detections)."
#     )