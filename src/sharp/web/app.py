import os
import io
import zipfile
import base64
import tempfile
from pathlib import Path

import requests
import gradio as gr

# Front-end Gradio app that calls the backend FastAPI service hosted on GPU cloud.
# Configure the backend base URL through environment variable on Hugging Face Spaces.
# Example: API_BASE_URL = "https://your-api.example.com"
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _files_payload(images):
    """Prepare multipart/form-data payload for requests.post(files=...)."""
    files = []
    for img in images:
        if img is None:
            continue
        # gr.Image(type="filepath") returns a string path
        if isinstance(img, str):
            path = img
            files.append(("files", (Path(path).name, open(path, "rb"), "image/*")))
            continue
        # gr.File returns objects with a .name attribute (path), or dict-like in some cases
        path = getattr(img, "name", None)
        if path is None and isinstance(img, dict) and "name" in img:
            path = img["name"]
        if path:
            files.append(("files", (Path(path).name, open(path, "rb"), "image/*")))
    return files


def predict_single(image):
    """Call /predict on backend for a single image and return one PLY file to download."""
    if not image:
        return None, "No image provided."
    files = _files_payload([image])
    if not files:
        return None, "Invalid image input."

    try:
        resp = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return None, f"Backend error: {e}"

    results = data.get("results", [])
    if not results:
        return None, "No result."
    item = results[0]
    if "error" in item:
        return None, item["error"]

    # Decode base64 PLY to a temporary file
    ply_bytes = base64.b64decode(item["ply_data"])
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmpf:
        tmpf.write(ply_bytes)
        ply_path = tmpf.name

    meta = f"{item['ply_filename']} ({item['width']}x{item['height']}), f={item['focal_length']:.2f}"
    return ply_path, meta


def predict_batch(images):
    """Call /predict on backend for multiple images and return a ZIP of PLY files."""
    if not images:
        return None, "No images provided."
    files = _files_payload(images)
    if not files:
        return None, "Invalid inputs."

    try:
        resp = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=300)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return None, f"Backend error: {e}"

    results = data.get("results", [])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        metas = []
        for item in results:
            if "error" in item:
                metas.append(f"{item.get('filename', '?')}: ERROR {item['error']}")
                continue
            ply_bytes = base64.b64decode(item["ply_data"])
            zf.writestr(item["ply_filename"], ply_bytes)
            metas.append(
                f"{item['filename']} -> {item['ply_filename']} "
                f"({item['width']}x{item['height']}, f={item['focal_length']:.2f})"
            )
    buf.seek(0)
    return buf, "\n".join(metas)


with gr.Blocks(title="SHARP View Synthesis") as demo:
    gr.Markdown(
        "# SHARP View Synthesis\nUpload image(s) to generate 3D Gaussian PLY files via the backend API."
    )

    with gr.Tab("Single Image"):
        in_img = gr.Image(type="filepath", label="Input Image")
        out_file = gr.File(label="Generated PLY")
        out_info = gr.Textbox(label="Info")
        btn = gr.Button("Predict")
        btn.click(predict_single, inputs=[in_img], outputs=[out_file, out_info])

    with gr.Tab("Batch"):
        in_imgs = gr.File(
            file_count="multiple", file_types=["image"], label="Input Images"
        )
        out_zip = gr.File(label="PLY ZIP")
        out_info2 = gr.Textbox(label="Info")
        btn2 = gr.Button("Predict Batch")
        btn2.click(predict_batch, inputs=[in_imgs], outputs=[out_zip, out_info2])

if __name__ == "__main__":
    # On Hugging Face Spaces, API_BASE_URL must point to your GPU cloud FastAPI server
    demo.launch(server_name="0.0.0.0", server_port=7860)
