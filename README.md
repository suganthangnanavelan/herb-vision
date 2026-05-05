# HerbVision

[![System Status](https://img.shields.io/website?down_color=red&down_message=offline&label=System%20Status&up_color=success&up_message=online&url=https%3A%2F%2Fherbvision.online%2Fhealth)](https://herbvision.online/health)
[![Model on HuggingFace](https://img.shields.io/badge/Hugging_Face-Model_Hub-orange)](https://huggingface.co/SuganthanGnanavelan/herbvision/tree/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-92.76%25-green)](#performance)

A deep learning system for identifying Traditional Chinese Medicine (TCM) herb species from photographs. HerbVision combines a Swin Transformer backbone with a weighted ensemble of eight specialist classifiers, achieving **92.76% validation accuracy** across **300 herb species** covering **52,089 training images**.

**Live demo: [herbvision.online](https://herbvision.online)**

---

## Table of Contents

- [Overview](#overview)
- [Dataset and Research](#dataset-and-research)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Medical Disclaimer](#medical-disclaimer)
- [License](#license)
- [Citation](#citation)

---

## Overview

Identifying dried and processed TCM herbs is a task that challenges even experienced practitioners. Many species share nearly identical visual characteristics after processing, and the same plant can look dramatically different depending on its preparation state. HerbVision addresses this through a three-stage hybrid intelligence pipeline that combines deep visual feature extraction with a specialist classifier ensemble and an alpha-fusion blend.

The system accepts a photograph — either uploaded from disk or captured via the live camera — and returns the top-5 most likely species alongside confidence scores, pharmacological metadata from a curated TCM database, and a reference image for visual cross-checking.

---

## Dataset and Research

### Dataset

The model was trained on **TCMP-300**, a curated dataset of 300 Traditional Chinese Medicine plant species with 52,089 training images.

**Dataset**: [https://doi.org/10.6084/m9.figshare.29432726](https://doi.org/10.6084/m9.figshare.29432726)

**Dataset Paper**: *TCMP-300: A benchmark dataset for Traditional Chinese Medicine plant identification.* Scientific Data (2025). [https://www.nature.com/articles/s41597-025-05522-7](https://www.nature.com/articles/s41597-025-05522-7)

### Base Paper

The methodology and results underlying HerbVision are described in detail in the following publication:

*A hybrid deep learning framework for Traditional Chinese Medicine plant recognition.* Frontiers in Plant Science (2025). [https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1672394/full](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1672394/full)

---

## Model Architecture

HerbVision uses a three-stage pipeline designed to handle the extreme inter-class visual similarity and varied processing states found in real-world TCM specimens.

### Stage 1 — Swin Transformer Backbone

The foundation is a `swin_base_patch4_window7_224` model pretrained on ImageNet-22K and fine-tuned on TCMP-300 using a two-phase strategy.

**Phase 1 — Warmup (4 epochs)**

Early transformer stages are frozen while only the classification head is trained. This stabilizes the head before full fine-tuning begins and prevents catastrophic forgetting of pretrained representations.

**Phase 2 — Full Fine-tuning**

All parameters are unfrozen. Training proceeds with a cosine annealing learning rate schedule, allowing the backbone to adapt its learned features to the domain of dried and processed botanical specimens.

A custom classification head replaces the default timm head:

```
LayerNorm(feat_dim)
Dropout(p=0.25)
Linear(feat_dim → 512)
GELU
Dropout(p=0.125)
Linear(512 → num_classes)
```

**Stage 1 result: 91.95% validation accuracy**

---

### Stage 2 — Specialist Ensemble

Ten-view Test-Time Augmentation (TTA) is applied at inference time to produce a **1024-dimensional feature descriptor** per image. This descriptor is compressed to **768 dimensions via PCA** and then passed to eight specialist classifiers trained to complement the transformer's decision boundary.

| Classifier | Role |
|---|---|
| SVM with RBF kernel | High-margin boundary separation |
| LightGBM | Gradient-boosted tree ensemble |
| XGBoost | Gradient-boosted tree ensemble |
| Logistic Regression | Linear baseline |
| Multi-Layer Perceptron | Non-linear learned features |
| Extra Trees | Randomised decision forest |
| Weighted soft-vote | Blended probabilistic consensus |
| Stacking meta-learner | Second-level learned blend |

---

### Stage 3 — Alpha-Fusion

An exhaustive sweep over blend coefficients identifies the optimal weighting between the Swin-B softmax outputs and the ensemble predictions.

**Optimal blend coefficient: alpha = 0.41**

```
final_prediction = (1 - alpha) * swin_probs + alpha * ensemble_probs
                 = 0.59 * swin_probs + 0.41 * ensemble_probs
```

**Stage 3 result: 92.76% validation accuracy**

---

## Performance

| Metric | Value |
|---|---|
| Validation Accuracy — Swin-B alone (Stage 1) | 91.95% |
| Validation Accuracy — Fused (Stage 3) | 92.76% |
| ROC AUC Score | 0.998 |
| Training Images | 52,089 |
| Herb Classes | 300 |
| Confidence Threshold for Unknown Detection | 0.20 |

The system flags any specimen with a top-1 confidence below **0.20** as unrecognised rather than forcing a low-confidence prediction onto the user.

---

## Features

**Plant Identification**

Upload a photograph from disk or use the live camera to classify a specimen against 300 TCM herb species. Results are returned in under one second on GPU and within a few seconds on CPU.

**Top-5 Predictions with Confidence**

The results panel displays the five most likely species, each with a confidence bar and score. Clicking any row updates the reference image to match that candidate, making it easy to compare alternatives.

**TCM Pharmacological Database**

Each identified herb is cross-referenced against a hand-curated `metadata.json` database providing:

- Chinese name and Pinyin romanisation
- Scientific and common names, family classification
- Thermal nature (cold, cool, neutral, warm, hot)
- Taste classification (bitter, sweet, pungent, salty, sour)
- Meridian affinity (which organ systems the herb is associated with)
- Parts used, pharmacological properties, description, and clinical uses

**Reference Image Comparison**

The interface displays a reference photograph of the top predicted species alongside the uploaded image. This visual cross-check helps users confirm or question the model's prediction before acting on it.

**Unknown Detection**

Specimens that fall below the confidence threshold are clearly flagged with a warning rather than silently returned as a low-confidence match.

**Live Camera Support**

A built-in camera modal allows capture directly from any device with a webcam or mobile camera, without requiring a separate app.

**Cold-Start Indicator**

On Railway deployments with a cold-start, the backend automatically polls `/health` every 7 seconds and surfaces a status banner until the model finishes loading from HuggingFace (~1–3 minutes). The interface becomes fully interactive the moment the backend is ready, with no manual refresh needed.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch, Swin-B via `timm` |
| Ensemble | scikit-learn, LightGBM, XGBoost |
| Backend | Flask, Flask-CORS |
| Frontend | Vanilla HTML/CSS/JS (single file, no framework) |
| Model Hosting | HuggingFace Hub |
| Deployment | Railway with custom domain via Namecheap |

---

## Project Structure

```
herbvision/
├── index.html          Single-page frontend (HTML, CSS, JS)
├── app.py              Flask backend, model loading, inference endpoints
├── metadata.json       TCM pharmacological database (300 species)
├── requirements.txt    Python dependencies
├── runtime.txt         Python version pin for Railway
├── Procfile            Process declaration for Railway deployment
├── LICENSE             MIT licence
├── README.md
├── assets/
│   └── preview.png     Application screenshot for documentation
└── reference/          Reference images shown for cross verification
```

On first startup, `app.py` downloads two files from HuggingFace and caches them locally in `hf_cache/`:

```
hf_cache/
├── class_names.json    List of 300 class label strings
└── swin_best.pth       Swin-B model checkpoint
```

---

## Installation

**Prerequisites:** Python 3.9 or later.

**1. Clone the repository**

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Start the server**

```bash
python app.py
```

On first run, the server will download `class_names.json` (11 KB) and `swin_best.pth` (333 MB) from HuggingFace and cache them in `hf_cache/`. Subsequent starts use the cache and load immediately.

**5. Open in browser**

```
http://localhost:5000
```

> **Note on GPU acceleration:** The backend automatically detects and uses CUDA if available. On CPU, inference takes a few seconds per image. On a CUDA-enabled GPU, inference is near-instant.

**Manual model download**

If automatic download fails (e.g. due to network restrictions), you can download the files manually from the [HuggingFace model page](https://huggingface.co/SuganthanGnanavelan/herbvision/tree/main) and place them in `hf_cache/`:

```
hf_cache/class_names.json
hf_cache/swin_best.pth
```

---

## Medical Disclaimer

> **The pharmacological profiles in `metadata.json` were generated with the assistance of AI and have not been independently verified against authoritative clinical or botanical sources. This information is intended for research and educational purposes only and must not be used for any medical or clinical decisions.**

---

## License

This project is licensed under the [MIT License](LICENSE).
