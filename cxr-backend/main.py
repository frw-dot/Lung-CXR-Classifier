"""
FastAPI backend for TB Chest X-Ray binary classification (M1b/M2).

Architecture: torchvision resnet50, fc replaced with nn.Linear(2048, 2)
Checkpoint format: state_dict only (torch.save(model.state_dict(), ...))

Run with:
    uvicorn main:app --reload --port 8000
"""

import io
import base64

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gradcam import GradCAM, overlay_heatmap


# ---------------------------------------------------------------------------
# CONFIG — edit these if your training setup differs
# ---------------------------------------------------------------------------

MODEL_PATH = "/Users/fadhillahrandywidiawan/Documents/Kerjaan/KODING/Prototipe 1/model/tb_model.pth"

# IMPORTANT: this order must match your training notebook's class_to_idx.
# If ImageFolder/your metadata-based Dataset was used, check the label
# mapping there ("Normal" -> 0, "TB" -> 1 in the current training script).
# Swap the two strings below if your predictions come out backwards.
CLASS_NAMES = ["Normal", "TB"]

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def load_model() -> nn.Module:
    model = build_model()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

# Last residual block (post-BN, post-skip-connection, post-ReLU) - the
# standard Grad-CAM target for ResNets. Targeting just .conv3 instead
# would hook the raw pre-BN/pre-residual conv output, which is noisier
# and less spatially coherent.
gradcam = GradCAM(model, target_layer=model.layer4[-1])

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------

app = FastAPI(title="TB CXR Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this once you deploy beyond localhost
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Uploaded file must be an image.")

    raw_bytes = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400, detail="Could not read image file.")

    # Resized RGB image (0-255 uint8) used as the base for the heatmap overlay
    display_img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    display_np = np.array(display_img).astype(np.float32) / 255.0  # HWC, 0-1

    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)  # 1xCxHxW

    # --- Inference + Grad-CAM in one pass ---
    logits, cam_map = gradcam(input_tensor)

    probs = F.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())

    overlay_img = overlay_heatmap(display_np, cam_map)  # HWC uint8 RGB

    buf = io.BytesIO()
    Image.fromarray(overlay_img).save(buf, format="PNG")
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "label": CLASS_NAMES[pred_idx],
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i].item()), 4)
            for i in range(len(CLASS_NAMES))
        },
        "gradcam_b64": gradcam_b64,
    })
