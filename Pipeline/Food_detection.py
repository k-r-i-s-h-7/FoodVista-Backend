



# ============================================================
# DETECT FOOD
# ============================================================


# from Working.Pipeline.Spoon_detection import IMAGE_PATH
from ultralytics import YOLO


class FoodDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_food(self, image_path):
        results = self.model(image_path)
        food_boxes = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                confidence = float(box.conf[0])

                # Ignore weak detections
                if confidence < 0.50:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                food_name = self.model.names[class_id]

                food_boxes.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class_id": class_id,
                    "name": food_name,
                    "confidence": confidence
                })

        if not food_boxes:
            raise RuntimeError("No food detected.")

        return food_boxes
    
def detect_food(model_path, image_path):
    
    food_model = YOLO(model_path)
    food_results = food_model(IMAGE_PATH)
    food_boxes = []

    for result in food_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(box.conf[0])

            # Ignore weak detections
            if confidence < 0.50:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_id = int(box.cls[0])

            food_name = food_model.names[class_id]

            food_boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_id": class_id,
                "name": food_name,
                "confidence": confidence
            })


    if not food_boxes:

        raise RuntimeError(
            "No food detected."
        )


    print("\n==============================")
    print("FOOD DETECTIONS")
    print("==============================")


    for i, food in enumerate(food_boxes):

        print(
            f"{i}: "
            f"{food['name']} "
            f"({food['confidence']:.3f})"
        )
    return food_boxes

