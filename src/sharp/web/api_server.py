import sys
import logging
import shutil
import tempfile
import zipfile
import io as python_io
import base64
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

# Ensure we can import the project package: add top-level 'src' to sys.path
# This file resides at: <repo_root>/src/sharp/web/api_server.py
# Path(__file__).parents[2] == <repo_root>/src
sys.path.append(str(Path(__file__).parents[2]))

from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
from sharp.utils import io as sharp_io
from sharp.utils.gaussians import save_ply
from sharp.cli.predict import predict_image, DEFAULT_MODEL_URL

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("sharp.api")

app = FastAPI()

# CORS - allow HF Spaces frontend to call this API.
# Consider tightening allow_origins to your Space domain for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: RGBGaussianPredictor | None = None
device: torch.device | None = None


@app.on_event("startup")
async def startup_event():
    global predictor, device
    try:
        device_str = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        device = torch.device(device_str)
        LOGGER.info(f"Using device: {device}")

        LOGGER.info("Loading SHARP model state dict...")
        state_dict = torch.hub.load_state_dict_from_url(
            DEFAULT_MODEL_URL, progress=True, map_location=device
        )

        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()
        predictor.to(device)
        LOGGER.info("Model loaded and ready.")
    except Exception as e:
        LOGGER.exception("Failed during startup/model init: %s", e)
        # Leave predictor as None; endpoints will return error until fixed.


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(device) if device else None,
        "model_loaded": predictor is not None,
    }


@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    """Accept images and return JSON with per-image metadata and PLY as base64."""
    if not predictor:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)

    results = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for file in files:
            try:
                # Persist upload to temp
                file_path = temp_path / file.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                # Load input and run prediction
                image, _, f_px = sharp_io.load_rgb(file_path)
                gaussians = predict_image(predictor, image, f_px, device)

                # Save PLY
                ply_filename = f"{file_path.stem}.ply"
                ply_path = temp_path / ply_filename
                height, width = image.shape[:2]
                save_ply(gaussians, f_px, (height, width), ply_path)

                # Encode PLY to base64 for transport
                with open(ply_path, "rb") as f:
                    ply_data = base64.b64encode(f.read()).decode("utf-8")

                results.append(
                    {
                        "filename": file.filename,
                        "ply_filename": ply_filename,
                        "ply_data": ply_data,
                        "width": width,
                        "height": height,
                        "focal_length": f_px,
                    }
                )
            except Exception as e:
                LOGGER.exception("Error processing %s: %s", file.filename, e)
                results.append({"filename": file.filename, "error": str(e)})

    return {"results": results}


@app.post("/predict/download")
async def predict_download(files: list[UploadFile] = File(...)):
    """Accept images and return a ZIP of generated PLY files."""
    if not predictor:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)

    output_zip = python_io.BytesIO()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(output_zip, "w") as zf:
            for file in files:
                try:
                    file_path = temp_path / file.filename
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)

                    image, _, f_px = sharp_io.load_rgb(file_path)
                    gaussians = predict_image(predictor, image, f_px, device)

                    ply_filename = f"{file_path.stem}.ply"
                    ply_path = temp_path / ply_filename
                    height, width = image.shape[:2]
                    save_ply(gaussians, f_px, (height, width), ply_path)

                    zf.write(ply_path, ply_filename)
                except Exception as e:
                    LOGGER.exception("Error processing %s: %s", file.filename, e)
                    continue

    output_zip.seek(0)
    return StreamingResponse(
        output_zip,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=gaussians.zip"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
