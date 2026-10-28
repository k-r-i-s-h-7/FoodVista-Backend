import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

class SpoonDetectionWithSAM:
    def __init__(self, model_path):
        self.model = SAM3SemanticPredictor(overrides={
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "model": model_path,
            "quantize": 16,
        })

    def detect_spoon(self, image_path):
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

        print(f"Image size: {image_width} x {image_height}")
        # print(f"Using device: {DEVICE}")
        self.model.set_image(image_path)
        results = self.model(text=["spoon"])
        if not results:
            raise RuntimeError("SAM 3 returned no results.")

        result = results[0]

        print("\nSAM 3 result:")
        print(result)

        if result.masks is None:
            raise RuntimeError(
                "SAM 3 did not generate any spoon masks."
            )

        masks = result.masks.data

        print(
            f"Number of masks detected: {len(masks)}"
        )
        
        masks_np = masks.detach().cpu().numpy()  # Select Spoon Mask

        mask_areas = [
            np.sum(mask > 0.5)
            for mask in masks_np
        ]

        largest_idx = np.argmax(mask_areas)

        spoon_mask = (
            masks_np[largest_idx] > 0.5
        )

        # Make absolutely sure dimensions match
        if spoon_mask.shape != (image_height, image_width):

            spoon_mask = cv2.resize(
                spoon_mask.astype(np.uint8),
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        print(
            f"Spoon mask area: "
            f"{np.sum(spoon_mask):,} pixels"
        )

        return results, spoon_mask


