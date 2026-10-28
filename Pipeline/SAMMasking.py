
# ============================================================
# RUN SAM 2
# ============================================================
import cv2
import numpy as np
from ultralytics import SAM
print("\nRunning SAM 2...")
class SAM2:
    def __init__(self, sam_model_path):
        self.sam_model = SAM(sam_model_path)
    def detect_food_area(self, image_path, results:dict):
            
            food_boxes = results["food_boxes"]
            spoon_width_px = results["spoon_width_px"]
            spoon_height_px = results["spoon_height_px"]
        
            image = cv2.imread(image_path)
        
            if image is None:
        
                raise RuntimeError(
                    f"Could not read image: {image_path}"
                )
        
            # RGB image for SAM
            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
            # ============================================================
            # SEGMENT EACH FOOD OBJECT
            # ============================================================
        
            segmented_foods = []
            for i, food in enumerate(food_boxes):
        
                box = [
                    food["x1"],
                    food["y1"],
                    food["x2"],
                    food["y2"]
                ]
                print(
                    f"\nSegmenting: "
                    f"{food['name']}"
                )
                # --------------------------------------------------------
                # SAM 2 prediction
                # --------------------------------------------------------
        
                sam_results = self.sam_model.predict(
                    source=image_path,
                    bboxes=[box],
                    verbose=False
                )
                if not sam_results:
                    print("SAM returned no result.")
                    continue
        
                sam_result = sam_results[0]
                if sam_result.masks is None:
        
                    print(
                        "SAM did not generate a mask."
                    )
        
                    continue
                # --------------------------------------------------------
                # Get masks
                # --------------------------------------------------------
        
                masks = sam_result.masks.data.cpu().numpy()
        
                if len(masks) == 0:
                    print("No mask generated.")
                    continue
        
                # SAM can generate multiple masks.
                # We select the mask with the largest overlap
                # with the YOLO bounding box.
        
                best_mask = None
                best_score = -1
        
                x1 = int(food["x1"])
                y1 = int(food["y1"])
                x2 = int(food["x2"])
                y2 = int(food["y2"])

                for mask in masks:
        
                    mask = mask.astype(np.uint8)
        
                    # Resize mask to original image dimensions
                    mask = cv2.resize(
                        mask,
                        (
                            image.shape[1],
                            image.shape[0]
                        ),
                        interpolation=cv2.INTER_NEAREST
                    )
                    # Calculate mask area
                    mask_area = np.sum(mask)

                    # Calculate mask inside YOLO bbox
                    bbox_region = mask[
                        y1:y2,
                        x1:x2
                    ]
                    bbox_area = max(
                        1,
                        (x2 - x1) * (y2 - y1)
                    )
        
                    overlap = np.sum(
                        bbox_region
                    )
                    score = overlap / bbox_area
                    if score > best_score:
                        best_score = score
                        best_mask = mask
        
        
                if best_mask is None:
        
                    continue
        
        
                # ========================================================
                # MASK MEASUREMENTS
                # ========================================================
        
                mask_area_px = int(
                    np.sum(best_mask)
                )
        
                bbox_area_px = (
                    (x2 - x1) *
                    (y2 - y1)
                )
        
        
                # Ratio of actual food pixels to bbox
                mask_bbox_ratio = (
                    mask_area_px /
                    max(1, bbox_area_px)
                )
        
                # ========================================================
                # NORMALIZE WITH SPOON
                # ========================================================
        
                food_width_px = x2 - x1
                food_height_px = y2 - y1
        
                food_area_spoon_ratio = (
                    mask_area_px /
                    (
                        spoon_width_px *
                        spoon_height_px
                    )
                )
        
                food_width_spoon_ratio = (
                    food_width_px /
                    spoon_width_px
                )
        
                food_height_spoon_ratio = (
                    food_height_px /
                    spoon_height_px
                )
        
                segmented_foods.append({
        
                    "name": food["name"],
        
                    "class_id": food["class_id"],
        
                    "confidence": food["confidence"],
        
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],
        
                    "mask": best_mask,
                    "mask_area_px": mask_area_px,
                    "bbox_area_px": bbox_area_px,
                    "mask_bbox_ratio": mask_bbox_ratio,
                    "food_width_px": food_width_px,
        
                    "food_height_px": food_height_px,
        
                    "food_area_spoon_ratio":
                        food_area_spoon_ratio,
        
                    "food_width_spoon_ratio":
                        food_width_spoon_ratio,
        
                    "food_height_spoon_ratio":
                        food_height_spoon_ratio
                })
            return segmented_foods
        
        
        
    
def run_sam2(sam_model, image_path, results:dict):
    food_boxes = results["food_boxes"]
    spoon_width_px = results["spoon_width_px"]
    spoon_height_px = results["spoon_height_px"]
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )
    # RGB image for SAM
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
    # ============================================================
    # SEGMENT EACH FOOD OBJECT
    # ============================================================
    segmented_foods = []
    for i, food in enumerate(food_boxes):
        box = [
            food["x1"],
            food["y1"],
            food["x2"],
            food["y2"]
        ]
        print(
            f"\nSegmenting: "
            f"{food['name']}"
        )
        # --------------------------------------------------------
        # SAM 2 prediction
        # --------------------------------------------------------
        sam_results = sam_model.predict(
            source=image_path,
            bboxes=[box],
            verbose=False
        )
        if not sam_results:
            print("SAM returned no result.")
            continue
        sam_result = sam_results[0]
        if sam_result.masks is None:
            print(
                "SAM did not generate a mask."
            )
            continue
        # --------------------------------------------------------
        # Get masks
        # --------------------------------------------------------
        masks = sam_result.masks.data.cpu().numpy()
        if len(masks) == 0:
            print("No mask generated.")
            continue
        # SAM can generate multiple masks.
        # We select the mask with the largest overlap
        # with the YOLO bounding box.
        best_mask = None
        best_score = -1
        x1 = int(food["x1"])
        y1 = int(food["y1"])
        x2 = int(food["x2"])
        y2 = int(food["y2"])
        for mask in masks:
            mask = mask.astype(np.uint8)
            # Resize mask to original image dimensions
            mask = cv2.resize(
                mask,
                (
                    image.shape[1],
                    image.shape[0]
                ),
                interpolation=cv2.INTER_NEAREST
            )
            # Calculate mask area
            mask_area = np.sum(mask)
            # Calculate mask inside YOLO bbox
            bbox_region = mask[
                y1:y2,
                x1:x2
            ]
            bbox_area = max(
                1,
                (x2 - x1) * (y2 - y1)
            )
            overlap = np.sum(
                bbox_region
            )
            score = overlap / bbox_area
            if score > best_score:
                best_score = score
                best_mask = mask
        if best_mask is None:
            continue
        # ========================================================
        # MASK MEASUREMENTS
        # ========================================================
        mask_area_px = int(
            np.sum(best_mask)
        )
        bbox_area_px = (
            (x2 - x1) *
            (y2 - y1)
        )
        # Ratio of actual food pixels to bbox
        mask_bbox_ratio = (
            mask_area_px /
            max(1, bbox_area_px)
        )
        # ========================================================
        # NORMALIZE WITH SPOON
        # ========================================================
        food_width_px = x2 - x1
        food_height_px = y2 - y1
        food_area_spoon_ratio = (
            mask_area_px /
            (
                spoon_width_px *
                spoon_height_px
            )
        )
        food_width_spoon_ratio = (
            food_width_px /
            spoon_width_px
        )
        food_height_spoon_ratio = (
            food_height_px /
            spoon_height_px
        )
        # ========================================================
        # SAVE DATA
        # ========================================================
        segmented_foods.append({
            "name": food["name"],
            "class_id": food["class_id"],
            "confidence": food["confidence"],
            "bbox": [
                x1,
                y1,
                x2,
                y2
            ],
            "mask": best_mask,
            "mask_area_px": mask_area_px,
            "bbox_area_px": bbox_area_px,
            "mask_bbox_ratio": mask_bbox_ratio,
            "food_width_px": food_width_px,
            "food_height_px": food_height_px,
            "food_area_spoon_ratio":
                food_area_spoon_ratio,
            "food_width_spoon_ratio":
                food_width_spoon_ratio,
            "food_height_spoon_ratio":
                food_height_spoon_ratio
        })
        print(
            f"Mask area: "
            f"{mask_area_px:,} px"
        )
        print(
            f"Mask/BBox ratio: "
            f"{mask_bbox_ratio:.3f}"
        )
        print(
            f"Food/Spoon area ratio: "
            f"{food_area_spoon_ratio:.3f}"
        )
