---
title: Exoplanet Transit Detection System
emoji: 🪐
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
license: mit
---

# 🪐 Exoplanet Transit Detection System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![NASA MAST](https://img.shields.io/badge/Data-NASA%20MAST-0B3D91?style=flat&logo=nasa&logoColor=white)](https://archive.stsci.edu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade AI system designed for the **ISRO Bharatiya Antariksh Hackathon (BAH) 2026** to detect exoplanet transit signals from noisy time-series stellar brightness data.

Features **1D Convolutional Neural Networks**, **1D Residual Networks (ResNet1D)**, **1D Grad-CAM Explainability Heatmaps**, **Box-Fitting Least Squares (BLS)** period searching, **Live NASA MAST Archive Integration**, **FastAPI REST Endpoints**, and an **Interactive Streamlit Web Dashboard**.

---

## 🌟 Key Features

* **Astronomical Signal Processing**:
  * Per-star Z-score normalization & SciPy 1D Gaussian noise smoothing ($\sigma = 1.0$).
  * **Phase Folding** over orbital periods ($P$) to align multi-transit dips.
  * **Box-Fitting Least Squares (BLS)** periodogram search for candidate period & transit epoch estimation.
  * NASA/Google AI **Dual View Representation** (Global View + Local Zoomed Transit Window).
* **Deep Learning Architecture**:
  * **3-Layer 1D CNN**: `Conv1D(16) -> Conv1D(32) -> Conv1D(64) -> Dense(64) -> Dropout(0.3) -> Sigmoid`.
  * **1D ResNet**: Residual skip connections to prevent gradient vanishing on long temporal sequences.
  * **SMOTE Class Balancing**: Resolves severe ~1.6% class imbalance (37 planet host stars vs. 2222 non-planet stars).
  * **Recall-Optimized Thresholding (0.30)**: Achieves **100% Planet Recall** with 0 false negatives.
* **Explainable AI (1D Grad-CAM)**:
  * Computes temporal gradient maps w.r.t. final Conv1D activations to highlight the exact transit dips driving prediction.
* **Live NASA Data Fetcher (`Lightkurve`)**:
  * Queries NASA MAST archive for live Kepler, K2, or TESS targets by ID (e.g. `KIC 10593626` / Kepler-22b).
* **Candidate Planet Radius Estimator**:
  * Calculates planet radius in Earth Radii ($R_{\oplus}$) from flux transit depth: $\frac{R_p}{R_*} = \sqrt{\delta}$.

---

## 🏗️ Repository Architecture

```
exoplanet-detection/
├── app.py                    # Hugging Face Spaces Streamlit entrypoint
├── app/
│   ├── dashboard.py          # Interactive Streamlit Web Application
│   └── main.py               # FastAPI REST API Server
├── src/
│   ├── data/
│   │   ├── loader.py         # Kepler dataset parser & synthetic data generator
│   │   └── nasa_api.py       # Live NASA MAST Lightkurve query integration
│   ├── features/
│   │   ├── preprocessing.py  # Normalization, Gaussian smoothing, outlier filtering
│   │   └── astronomy.py      # BLS search, phase-folding, local/global views, R_p estimator
│   ├── models/
│   │   ├── cnn1d.py          # 1D CNN architecture builder
│   │   ├── resnet1d.py       # 1D Residual Network (ResNet1D) architecture
│   │   ├── local_global_cnn.py# Dual-branch local + global CNN
│   │   └── explainability.py # 1D Grad-CAM attention heatmap generator
│   └── utils/
│       └── metrics.py        # Recall-focused metrics, ROC-AUC, threshold optimizer
├── train.py                  # CLI training script
├── evaluate.py               # CLI evaluation script
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker containerization config
└── README.md
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites & Environment Setup
Clone the repository and set up a Python 3.11 virtual environment using `uv` or `venv`:

```bash
git clone https://github.com/anuragkkumar/exoplanet-detection.git
cd exoplanet-detection

# Create & activate virtual environment with uv
uv venv .venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

---

## 🚀 Running the Web Dashboard & REST API

### Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. Features include:
* Interactive Plotly Light Curve charts (Raw vs Smoothed).
* 1D Grad-CAM Attention Heatmaps overlaid on detected transit dips.
* Live NASA Kepler / TESS ID lookup (e.g. `KIC 10593626`).
* Custom CSV / TXT light curve upload.
* Phase Folding & BLS period parameter controls.
* Candidate Planet Radius ($R_{\oplus}$) estimation.

### Launch FastAPI REST Server
```bash
uvicorn app.main:app --reload --port 8000
```
Interactive Swagger API documentation available at `http://localhost:8000/docs`.

---

## 📊 Benchmark Results

| Model Architecture | Class Balance | Decision Threshold | Overall Accuracy | Planet Recall | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1D CNN** | Class Weights | 0.50 | 99.1% | 81.0% | 0.962 |
| **SMOTE + 1D CNN** | SMOTE | 0.50 | 99.2% | 94.6% | 0.978 |
| **Tuned 3-Layer CNN (Final)** | **SMOTE** | **0.30** | **99.3%** | **100.0%** | **0.984** |

---

## 📜 Built For
Developed for the **ISRO Bharatiya Antariksh Hackathon (BAH) 2026**.
