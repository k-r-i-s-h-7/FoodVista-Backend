
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from FoodAreaPipeline import FoodAreaPipeline


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "Metrics"

MAX_IMAGE_SIZE = 15 * 1024 * 1024  # 15 MB

ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}


def get_path(env_name: str, default_path: Path) -> str:
    """Allow deployment paths to be overridden by environment variables."""
    return os.getenv(env_name, str(default_path))


def build_pipeline() -> FoodAreaPipeline:
    """Load all models once when the API starts."""
    return FoodAreaPipeline(
        food_model_path=get_path(
            "FOOD_MODEL_PATH",
            MODELS_DIR  / "classif.pt",
        ),
        utensil_model_path=get_path(
            "UTENSIL_MODEL_PATH",
            MODELS_DIR / "spoon_model.pt",
        ),
        sam_model_path=get_path(
            "SAM_MODEL_PATH",
            MODELS_DIR / "sam3.pt",
        ),
        depth_api_url=os.getenv(
            "DEPTH_API_URL",
            "http://localhost:8000/predict-depth-map",
        ),
        food_confidence=0.50,
        utensil_confidence=0.25,
        ingredient_model_path=get_path(
            "INGREDIENT_MODEL_PATH",
            MODELS_DIR / "ing_model"
            ".pt",
        ),
        density_data_path=get_path(
            "DENSITY_DATA_PATH",
            CONFIG_DIR / "Density.json",
        ),
        density_classes_path=get_path(
            "DENSITY_CLASSES_PATH",
            CONFIG_DIR / "DensityRange.json",
        ),
        nutrition_db_path=get_path(
            "NUTRITION_DB_PATH",
            CONFIG_DIR / "Ingredients.json",
        ),
        single_items_path=get_path(
            "SINGLE_ITEMS_PATH",
            CONFIG_DIR / "SingleItems.json",
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail startup if the model files or dependencies are invalid.
    app.state.pipeline = build_pipeline()
    yield


app = FastAPI(
    title="Food Estimation API",
    version="1.0.0",
    lifespan=lifespan,
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def make_json_safe(value):
    """Convert NumPy values into JSON-compatible output."""
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        # Avoid returning huge segmentation masks and depth maps.
        if value.ndim >= 2:
            summary = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }

            if value.size:
                summary["min"] = float(np.nanmin(value))
                summary["max"] = float(np.nanmax(value))

            return summary

        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    return value


@app.get("/health")
async def health():
    return {"status": "ok"}


from fastapi import Request


from fastapi import Request, HTTPException
from starlette.datastructures import UploadFile as StarletteUploadFile


@app.post("/predict")
async def predict(request: Request):
    try:
        form = await request.form()
        print("CONTENT TYPE:", request.headers.get("content-type"))
        print("FORM KEYS:", list(form.keys()))

        image = form.get("image")

        # image = form.get("image")

        print("IMAGE TYPE:", type(image))
        print("IMAGE FILENAME:", getattr(image, "filename", None))
        print("IMAGE CONTENT TYPE:", getattr(image, "content_type", None))
        print("IMAGE HAS READ:", hasattr(image, "read"))
        filename = getattr(image, "filename", None) or "upload.jpg"
        suffix = Path(filename).suffix.lower()

        if suffix not in ALLOWED_EXTENSIONS:
            suffix = ".jpg"

        contents = await image.read(MAX_IMAGE_SIZE + 1)

        if not contents:
            raise HTTPException(status_code=400, detail="Empty upload")

        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="Image exceeds 15 MB")

        decoded = cv2.imdecode(
            np.frombuffer(contents, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if decoded is None:
            raise HTTPException(
                status_code=400,
                detail="Uploaded bytes could not be decoded as an image",
            )

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temp_file:
                temp_file.write(contents)
                temp_path = temp_file.name

            result = await run_in_threadpool(
                request.app.state.pipeline.estimate,
                temp_path,
                True,
            )

            return {
                "success": True,
                "filename": filename,
                "result": make_json_safe(result),
            }

        finally:
            await image.close()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    except HTTPException:
        raise
    except Exception as exc:
        import logging
        logging.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
    )