# Lung-CXR-Classifier

A deep-learning prototype for **tuberculosis (TB) screening on chest radiographs (CXR)**. The current model performs binary classification — **Normal vs TB** — using a ResNet50 convolutional neural network fine-tuned on a combined, multi-source CXR dataset.

This repository is part of an ongoing clinical–AI research effort aimed at supporting radiographic triage in resource-limited settings, where TB burden is high and expert radiology reads are not always immediately available.

> ⚠️ **Research / educational use only.** This model is a prototype and is **not** validated for clinical decision-making or patient care.

---

## Overview

| | |
|---|---|
| **Task** | Binary CXR classification (Normal / TB) |
| **Architecture** | ResNet50 (ImageNet-pretrained, fine-tuned) |
| **Framework** | PyTorch + torchvision |
| **Input** | Chest X-ray images (PNG) |
| **Image size** | 224 × 224 |
| **Output** | Class label + TB probability |

---

## Clinical motivation

Tuberculosis remains a major public-health challenge, and chest radiography is a first-line screening tool. An automated triage model that flags likely-abnormal films can help prioritize review, reduce time-to-diagnosis, and extend screening reach where radiologist availability is limited. This prototype is an early step toward that goal.

---

## Repository contents

```
Lung-CXR-Classifier/
├── lung-tb-cxr-classifier.ipynb   # End-to-end training & evaluation notebook
├── CXR Dataset Multiple Sources   # Notes on the combined dataset sources
├── cxr-backend/                   # FastAPI + Grad-CAM web app for serving the model
└── README.md
```

---

## Method

The notebook (`lung-tb-cxr-classifier.ipynb`) runs the full pipeline end to end.

**Preprocessing**
- Convert to grayscale, then expand to 3 channels (for ImageNet-pretrained backbone)
- Resize to 224 × 224
- Normalize with ImageNet mean/std

**Data handling**
- Metadata built from a `Normal/` and `TB/` folder structure
- `source` parsed from each filename prefix to track dataset origin
- **Stratified** 80/20 train/validation split by `label + source`, so every source is represented proportionally in both sets

**Class imbalance**
- Class weights computed from training-set frequencies and passed to `CrossEntropyLoss`, mitigating the TB : Normal imbalance

**Model & training**
- `resnet50(weights=ResNet50_Weights.DEFAULT)` with the final FC layer replaced by a 2-class linear head
- Optimizer: Adam (`lr = 0.001`)
- Loss: weighted cross-entropy
- Batch size: 32 · Epochs: 10 · Seed: 42 (reproducibility)
- Trained weights saved to `tb_model.pth`

**Evaluation**
- Validation accuracy per epoch
- Confusion matrix
- **Sensitivity** and **specificity** (clinically relevant for a screening tool)
- **AUC** and ROC curve

---

## Dataset

A combined chest X-ray dataset assembled from multiple public sources (see `CXR Dataset Multiple Sources`). Images are organized by class:

```
combined_cxr_dataset/
├── Normal/
└── TB/
```

Filenames are prefixed with their source identifier so that stratification preserves the source mix across train/validation splits.

> Dataset files are **not** committed to this repository. Update `ROOT_DIR` in the notebook to point to your local or Kaggle dataset path.

---

## Getting started

### Requirements

```bash
pip install torch torchvision scikit-learn pandas numpy pillow matplotlib
```

A CUDA-capable GPU is recommended (the notebook auto-detects and falls back to CPU).

### Run

1. Open `lung-tb-cxr-classifier.ipynb` (locally or on [Kaggle](https://www.kaggle.com/code/fadhillahrandyw/lung-tb-cxr-classifier)).
2. Set `ROOT_DIR` / `METADATA_CSV` to your dataset location.
3. Run all cells to train, evaluate, and export `tb_model.pth`.

---

## Deployment / web app

`cxr-backend/` is a local web app for serving the trained model: a FastAPI backend
(`/predict`, `/health`) that runs inference with the exported `tb_model.pth` and returns
a Grad-CAM heatmap alongside the prediction, plus a plain HTML/JS frontend. See
[`cxr-backend/README.md`](cxr-backend/README.md) for setup and usage.

---

## Results

*Metrics on the held-out validation set:*

| Metric | Value |
|---|---|
| Accuracy | 93,5% |
| Sensitivity (recall for TB) | 94,57% |
| Specificity | 90,19% |
| AUC | 97,81% |

For a screening model, **sensitivity** (minimizing missed TB cases) is the priority metric, read alongside specificity to gauge the false-positive burden.

---

## Roadmap

- [ ] Move from binary (Normal / TB) to **multiclass** CXR classification (e.g. Normal / TB / Pneumonia / other)
- [ ] Validate on a **prospective, locally-sourced dataset**
- [x] Add explainability (e.g. Grad-CAM heatmaps) to support radiologist review — see `cxr-backend/`
- [ ] External validation across scanners/sites to test generalization
- [ ] Prototype integration into an emergency/triage workflow

---

## Limitations

- Trained and validated on retrospective, largely public data — performance may not transfer to a different scanner, population, or acquisition protocol.
- Binary output only; it does not localize findings or distinguish TB from other abnormalities.
- **Not a medical device.** Any clinical use would require prospective validation, regulatory review, and radiologist oversight.

---

## Author

**Fadhillah Randy Widiawan (frw-dot)** — physician (Indonesia).

## License

No license specified yet
