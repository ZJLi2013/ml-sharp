# SHARP Web: Frontend (Gradio) + Backend (FastAPI)

This directory provides a separated deployment:
- Backend API (`api_server.py`) runs on a GPU cloud with FastAPI
- Frontend UI (`app.py`) runs on Hugging Face Spaces with Gradio

The UI calls the API via HTTP; the API performs model inference and returns PLY results.

## Repository Layout

- `src/sharp/web/api_server.py` — FastAPI backend hosting inference endpoints
- `src/sharp/web/app.py` — Gradio frontend calling the backend
- `requirements_api.txt` — Backend dependencies (GPU cloud)
- `requirements.txt` — Frontend dependencies (HF Spaces)

## Backend (GPU Cloud)

### Install

On your GPU cloud instance:
```bash
# From repository root
pip install -r requirements_api.txt
```

Notes:
- Ensure CUDA is available if using NVIDIA GPUs. The Torch version in `requirements_api.txt` is compiled for CUDA 12 on Linux.
- On macOS, MPS (Apple Silicon) may be detected; otherwise CPU fallback is used.

### Run

From repository root:
```bash
python src/sharp/web/api_server.py
```
or with Uvicorn:
```bash
uvicorn src.sharp.web.api_server:app --host 0.0.0.0 --port 8000
```

### Endpoints

- `GET /health` — Basic health check, device info, and model-loaded flag
- `POST /predict` — Multipart upload of one or more images (`files` field); returns JSON with per-image metadata and PLY contents base64-encoded
- `POST /predict/download` — Multipart upload of one or more images; returns a ZIP stream containing PLY files

CORS is enabled by default to allow calls from the Hugging Face Space. For production, set `allow_origins` to your specific Space domain.

## Frontend (Hugging Face Spaces)

### Install

On HF Spaces:
```bash
pip install -r requirements.txt
```
This installs only Gradio and Requests.

### Configure

Set environment variable `API_BASE_URL` in your Space to point to the public backend URL, for example:
```
API_BASE_URL=https://your-api.example.com
```
If running locally for testing, `API_BASE_URL` defaults to `http://localhost:8000`.

### Run

Locally:
```bash
python src/sharp/web/app.py
```
Gradio will start on port `7860` by default (configured to `0.0.0.0` in the script).

On HF Spaces, simply setting the Space’s “Run” command to `python src/sharp/web/app.py` is sufficient.

### Usage (Frontend)

- Single Image tab: upload one image and click Predict to download its PLY.
- Batch tab: upload multiple images and click Predict Batch to download a ZIP containing PLY files.
- The frontend calls the backend `POST /predict` and assembles results for user download.

## Quick Local Test

1. Start backend:
   ```bash
   uvicorn src.sharp.web.api_server:app --host 0.0.0.0 --port 8000
   ```

2. Start frontend (in another terminal):
   ```bash
   API_BASE_URL=http://localhost:8000 python src/sharp/web/app.py
   ```

3. Open the Gradio UI (http://localhost:7860), upload images, and verify outputs.

## Notes & Troubleshooting

- If imports like `fastapi` or `gradio` show unresolved in your IDE, ensure the correct environment is selected and dependencies installed via the respective requirements file.
- Network access from HF Spaces to the GPU API must be allowed; ensure your API endpoint is accessible over HTTPS where possible.
- For security, consider locking down CORS to your Space origin and adding authentication (e.g., an API key header) if needed.
