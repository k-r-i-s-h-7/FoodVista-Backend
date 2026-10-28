

import torch

class MiDaSDepthEstimator:
    def __init__(self, model_name="DPT_Hybrid", device=None):
        self.model_name = model_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.transform = self.load_transform()

    def load_model(self):
        print("\nLoading MiDaS...")
        model = torch.hub.load("intel-isl/MiDaS", self.model_name)
        model.to(self.device)
        model.eval()
        return model

    def load_transform(self):
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_name in ["DPT_Large", "DPT_Hybrid"]:
            return midas_transforms.dpt_transform
        else:
            return midas_transforms.small_transform

    def get_depth_map(self,image_rgb):
        with torch.no_grad():
            input_batch = self.transform(image_rgb).to(self.device)
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth_map = prediction.cpu().numpy()
            return depth_map

    

def get_depth_map(image_rgb, MIDAS_MODEL, DEVICE):
    print("\nLoading MiDaS...")

    midas = torch.hub.load(
        "intel-isl/MiDaS",
        MIDAS_MODEL
    )

    midas.to(DEVICE)
    midas.eval()


    # ============================================================
    # 5. LOAD MiDaS TRANSFORM
    # ============================================================

    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms"
    )

    if MIDAS_MODEL in [
        "DPT_Large",
        "DPT_Hybrid"
    ]:

        transform = (
            midas_transforms.dpt_transform
        )

    else:

        transform = (
            midas_transforms.small_transform
        )


    # ============================================================
    # 6. PREPROCESS IMAGE FOR MiDaS
    # ============================================================

    input_batch = transform(
        image_rgb
    )

    input_batch = input_batch.to(
        DEVICE
    )


    # ============================================================
    # 7. MiDaS DEPTH INFERENCE
    # ============================================================

    print("Running MiDaS...")

    with torch.no_grad():

        prediction = midas(
            input_batch
        )

        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
        )


    # ============================================================
    # 8. GET DEPTH MAP
    # ============================================================

    depth = prediction.cpu().numpy()

    print(
        f"\nDepth map shape: {depth.shape}"
    )

    print(
        f"Global depth min: {depth.min():.6f}"
    )

    print(
        f"Global depth max: {depth.max():.6f}"
    )