# HerbVision

**TCM Plant Intelligence — TCMP-300 Pipeline v3**

A deep learning system for identifying Traditional Chinese Medicine (TCM) herb species from photographs. HerbVision combines a Swin Transformer backbone with a weighted ensemble of eight specialist classifiers, achieving 92.76% validation accuracy across 300 herb species.

**Live Demo:** [https://www.herbvision.online](https://www.herbvision.online)

---

## Preview

![HerbVision Homepage](assets/preview.png)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Inference Pipeline](#inference-pipeline)
- [Performance](#performance)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## Overview

HerbVision is a two-stage hybrid intelligence pipeline built for real-world TCM herb identification. The system handles the hardest challenges in this domain — extreme inter-class visual similarity, varied processing states (raw root, dried bark, powdered form), and diverse imaging conditions — and returns both a ranked classification and a full pharmacological profile for each identified species.

The model was trained on the TCMP-300 benchmark and is served through a Flask backend with a browser-based frontend that supports photo upload and live camera capture.

---

## Dataset

| Attribute | Detail |
|---|---|
| Name | TCMP-300 |
| Species | 300 TCM herb classes |
| Total Images | 52,089 expert-verified images |
| Source | Kaggle |
| Link | [https://www.kaggle.com/datasets/suganthang/tcmp-300](https://www.kaggle.com/datasets/suganthang/tcmp-300) |

Images cover diverse specimen forms including roots, leaves, flowers, bark, and processed medicinal forms. All images are expert-verified for label correctness.

---

## Model Architecture

HerbVision uses a three-stage pipeline: a Transformer backbone for feature extraction, a specialist ensemble for secondary classification, and an optimal blend layer that fuses both outputs.

### Stage 01 — Swin Transformer Backbone

A `swin_base_patch4_window7_224` model pretrained on ImageNet-22K (via `timm`) is fine-tuned using a two-phase strategy:

- **Phase 1 (Warmup):** Four epochs with early stages frozen, allowing the classification head to stabilise before full training begins.
- **Phase 2 (Full Fine-tuning):** All parameters unfrozen. Cosine annealing schedule applied with layer-wise learning rate differentiation — lower layers receive smaller learning rates to preserve low-level features.

Additional training techniques include HybridMix augmentation, drop path regularisation (`rate=0.15`), and a custom head with LayerNorm, GELU activations, and dropout (`p=0.25`).

**Stage 01 best validation accuracy: 91.95%**

### Stage 02 — Specialist Ensemble

Ten-view Test-Time Augmentation (TTA) produces a 1024-dimensional feature descriptor per image. This descriptor is compressed via PCA to 768 dimensions (retaining at least 95% of variance) and passed to eight specialist classifiers trained under 5-fold stratified cross-validation:

- SVM with RBF kernel
- LightGBM
- XGBoost
- Logistic Regression
- Multi-Layer Perceptron
- Extra Trees
- Weighted soft-vote ensemble
- Stacking meta-learner

### Stage 03 — Optimal Blend (α-Fusion)

An exhaustive sweep across 91 mixing ratios identifies the optimal blend coefficient at **α = 0.41**, combining Swin-B softmax outputs with ensemble predictions. The fusion strategy is validated via cross-validation, with gains concentrated in historically confusable herb classes.

**Final fused accuracy: 92.76% (+0.81% over backbone alone)**

---

## Inference Pipeline

```
Image Input (224x224)
        |
        v
  Swin-B Backbone
  (1024-dim features)
        |
        v
  10-View TTA
  (averaged pooling)
        |
        v
  PCA Compression
  (768-dim, 95% variance)
        |
        v
  Ensemble of 8 Classifiers
        |
        v
  Alpha-Blend (α = 0.41)
  Swin output + Ensemble output
        |
        v
  Classification Result
  (300 classes, Top-5 with confidence)
```

---

## Performance

| Metric | Value |
|---|---|
| Validation Accuracy (Swin-B alone) | 91.95% |
| Validation Accuracy (Fused) | 92.76% |
| ROC AUC Score | 0.998 |
| Training Images | 52,089 |
| Herb Classes | 300 |
| Ensemble Classifiers | 8 |
| Optimal Blend Coefficient (α) | 0.41 |
| Confidence Threshold | 0.20 |

---

## Features

**Plant Identification**
Upload a photograph or use the live camera to identify a TCM herb. The system returns a ranked list of up to five candidate species with individual confidence scores.

**Full TCM Pharmacological Database**
Every identified species is paired with a complete TCM profile retrieved from `metadata.json`, including:

- Chinese name and Pinyin romanisation
- Scientific name and botanical family
- Thermal nature (Hot / Warm / Neutral / Cool / Cold)
- Taste classification
- Meridian affinity
- Parts used
- Pharmacological properties
- Clinical description and traditional uses

**Reference Specimen Comparison**
After analysis, a reference image of the top-predicted species is displayed side-by-side with the uploaded photo, allowing visual cross-verification.

**Unknown Detection**
Specimens that fall below the confidence threshold (0.20) are flagged as unrecognised rather than being assigned a false classification.

**Top-5 Candidate List**
All five ranked candidates are interactive. Clicking any candidate updates the pharmacological panel and reference image, which is useful for morphologically ambiguous specimens.

**Live Camera Capture**
Real-time camera input is supported via the browser. Works with mobile rear cameras and desktop webcams.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning Framework | PyTorch |
| Backbone | Swin-B via `timm` |
| Ensemble Classifiers | scikit-learn, LightGBM, XGBoost |
| Backend | Flask, Flask-CORS |
| Image Processing | Pillow, torchvision |
| Model Hosting | HuggingFace Hub (`SuganthanGnanavelan/herbvision`) |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Deployment | https://www.herbvision.online |

---

## Project Structure

```
herbvision/
├── app.py                  # Flask backend — model loading, prediction endpoint
├── index.html              # Frontend — single-page interface
├── requirements.txt        # Python dependencies
├── metadata.json           # TCM pharmacological database (300 species)
├── hf_cache/               # Local cache for HuggingFace downloads
│   ├── class_names.json    # Class index to herb name mapping
│   └── swin_best.pth       # Model checkpoint
├── reference/              # Reference specimen images served by /reference/<classname>
│   └── <class_name>.jpg
└── sample_images/          # Example herb photographs for testing
    └── <herb_name>.jpg
```

---

## Installation

**Prerequisites:** Python 3.9 or later, pip

```bash
# Clone the repository
git clone https://github.com/your-username/herbvision.git
cd herbvision

# Install dependencies
pip install -r requirements.txt
```

On first run, the application automatically downloads `class_names.json` and `swin_best.pth` from HuggingFace and caches them in `hf_cache/`. An active internet connection is required for the initial download.

---

## Usage

```bash
python app.py
```

The server starts on `http://0.0.0.0:5000`. Open `http://localhost:5000` in a browser to access the interface.

To use the deployed version directly, visit [https://www.herbvision.online](https://www.herbvision.online).

### Steps

1. Navigate to the **Analyzer** section.
2. Upload a plant photograph (root, leaf, flower, bark, or processed form) or use **Live Camera** for direct capture.
3. Click **Unmask This Plant**.
4. View the confidence score, top-5 candidate list, reference specimen image, and full TCM pharmacological details.

---

## License

This project is licensed under the [MIT License](LICENSE).

The TCMP-300 dataset is separately licensed on Kaggle. Please review the dataset terms before any commercial or clinical application.

---

*HerbVision · TCMP-300 Pipeline v3 · Swin-B + Weighted Ensemble · α = 0.41*
