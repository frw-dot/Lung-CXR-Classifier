# TB CXR Triage — Local Web App (M1b)

A minimal local web app for your TB chest X-ray classifier: FastAPI backend
(PyTorch ResNet50 + Grad-CAM) + a plain HTML/JS frontend.

This is the deployment layer for the training notebook in the parent
[Lung-CXR-Classifier](..) repo — see the root [README](../README.md) for the
model architecture, dataset, and training details.

## What's inside

```
cxr-backend/
├── main.py              FastAPI app: /predict and /health endpoints
├── gradcam.py            Self-contained Grad-CAM implementation
├── requirements.txt       Python dependencies
├── model/
│   └── tb_model.pth       Your trained weights (already copied in)
└── index.html             Frontend — open directly in a browser
```

## 1. Install dependencies

```bash
cd cxr-backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. IMPORTANT — verify class order before running

Open `main.py` and check this line near the top:

```python
CLASS_NAMES = ["Normal", "TB"]
```

This must match the index order your model was trained with. In your Kaggle
notebook, find and run:

```python
print(train_dataset.class_to_idx)
```

If it prints `{'TB': 0, 'Normal': 1}`, flip the list to:

```python
CLASS_NAMES = ["TB", "Normal"]
```

If this is wrong, the app will still run without errors — it will just
label TB cases as Normal and vice versa, so it's worth double-checking
before you trust any results.

## 3. Run the backend

```bash
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/health` in your browser — you should see:

```json
{"status": "ok", "device": "cpu", "classes": ["Normal", "TB"]}
```

(`device` will say `cuda` instead of `cpu` if you have a GPU set up locally.)

## 4. Open the frontend

Just open `index.html` directly in your browser (double-click it, or
`open index.html` / `start index.html`). No server needed for the frontend
itself — it's a static file that calls `localhost:8000/predict`.

Upload a chest X-ray, click **Run inference**, and you'll see:
- The predicted label (Normal / TB) with confidence
- A side-by-side Grad-CAM heatmap overlay
- The full probability breakdown for both classes

## Notes on the model

Your checkpoint is a standard `torchvision.models.resnet50` with the final
`fc` layer replaced by `nn.Linear(2048, 2)` — matching the training notebook
in the parent repo. No custom layers, so `build_model()` in `main.py`
reconstructs the exact architecture before loading weights.

Grad-CAM is computed on `layer4[-1]`, the last residual block before global
average pooling — the standard choice for ResNet-style CNNs, since it has
the best balance of spatial resolution and class-discriminative features.

## Troubleshooting

**"Could not reach the backend"** in the browser — make sure `uvicorn` is
still running in your terminal and that you're using port 8000 (or update
`API_URL` at the top of `index.html`'s `<script>` block if you changed it).

**CORS errors in the browser console** — the backend already allows all
origins (`allow_origins=["*"]`) for local development, so this shouldn't
happen with `index.html` opened locally. If it does, try serving the
frontend instead of opening it as a `file://` URL:

```bash
python3 -m http.server 5500
```

then visit `http://localhost:5500/index.html`.

**Model fails to load / shape mismatch** — this would mean the checkpoint
doesn't match `build_model()`'s architecture. Since this was verified
directly from your uploaded `.pth` file, it should load cleanly as-is.
