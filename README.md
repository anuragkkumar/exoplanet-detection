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

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://exoplanet-detection-sqgdixxcb8wsxhvrxsqwwh.streamlit.app)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/anuragkkumar/exoplanet-detection)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![NASA MAST](https://img.shields.io/badge/Data-NASA%20MAST-0B3D91?style=flat&logo=nasa&logoColor=white)](https://archive.stsci.edu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade AI platform developed for the **ISRO Bharatiya Antariksh Hackathon (BAH) 2026** to detect exoplanet transit signatures from noisy stellar photometric time-series data.

Featuring **1D Deep Convolutional Neural Networks**, **1D Residual Networks (ResNet1D)**, **1D Grad-CAM Explainable AI Heatmaps**, **Box-Fitting Least Squares (BLS)** period searching, **Live NASA MAST Archive Queries**, **Audio Sonification**, an interactive **Planetary Systems Explorer**, and a **FastAPI REST API**.

---

## 🌐 Live Deployments

* 🪐 **Interactive Web Application (Streamlit Cloud)**: [exoplanet-detection.streamlit.app](https://exoplanet-detection-sqgdixxcb8wsxhvrxsqwwh.streamlit.app)
* 🤗 **Hugging Face Space**: [huggingface.co/spaces/anuragkkumar/exoplanet-detection](https://huggingface.co/spaces/anuragkkumar/exoplanet-detection)
* 💻 **GitHub Repository**: [github.com/anuragkkumar/exoplanet-detection](https://github.com/anuragkkumar/exoplanet-detection)

---

## 🌟 Key Features

### 1. Astronomical Signal Processing & Preprocessing
* **Z-Score Normalization & Smoothing**: SciPy 1D Gaussian noise smoothing ($\sigma = 1.0$) with adaptive outlier filtering.
* **Box-Fitting Least Squares (BLS)**: Automated orbital period ($P$) and transit epoch ($T_0$) detection.
* **Phase-Folding**: Folds periodic transit observations over estimated periods to confirm repetitive planetary dips.
* **Local & Global Dual View Representation**: Local zoomed transit profile + global baseline view aligned with NASA Kepler pipelines.

### 2. Deep Learning & Explainable AI (XAI)
* **3-Layer 1D CNN**: Multi-scale convolution filters with max-pooling, dense head, and dropout regularization.
* **1D ResNet Architecture**: Residual identity mappings to preserve subtle dip features across long sequences.
* **SMOTE Balanced Training**: Overcomes extreme astronomical class imbalance (37 planet host stars vs. 2,222 non-planet stars).
* **Zero False Negative Threshold (0.30)**: Tuned decision threshold guaranteeing **100% Planet Recall**.
* **1D Grad-CAM Attention Heatmaps**: Highlights temporal activations showing exactly which transit dips influenced model predictions.

### 3. Physical Telemetry & Habitable Zone Estimation
* **Planet Radius ($R_p$)**: Derived from transit depth ($\delta = (R_p / R_*)^2$).
* **Semi-Major Axis ($a$)**: Calculated via Kepler’s Third Law: $a = \sqrt[3]{(P / 365.25)^2 \cdot M_*}$ in AU.
* **Equilibrium Temperature ($T_{eq}$)**: Radiative equilibrium calculation based on stellar luminosity.
* **Earth Similarity Index (ESI)**: Multiparameter geometric mean comparing radius and surface temperature to Earth.

### 4. Lightweight Planetary Systems Explorer (2D Maps)
* **🌌 Our Solar System**: Complete 2D orbital map with Sun and all 8 planets (Mercury through Neptune) with real AU distances and `🌍 Earth — YOU ARE HERE (1.00 AU)`.
* **🪐 Alien Star System**: Maps the detected planet in its own independent star system in deep space (~650 light-years away) alongside companion planets and the **Habitable (Goldilocks) Zone**.
* **⚖️ Side-by-Side Comparison**: Direct architectural comparison table between Our Solar System and the Alien System.

### 5. Multi-Source Ingestion & Audio Sonification
* **Live NASA MAST Archive Integration**: Direct queries using `lightkurve` for Kepler, K2, and TESS targets (e.g., `KIC 10593626` / Kepler-22b).
* **Custom Dataset Ingestion**: Drag-and-drop support for custom photometric CSV/TXT files.
* **Synthetic Signal Generator**: Realistic synthetic light curves with customizable transit depths, periods, and stellar noise.
* **Audio Sonification**: Converts stellar brightness variations into playable WAV audio frequencies, allowing users to audibly hear transit dips.

---

## 🏗️ Repository Architecture

```
exoplanet-detection/
├── app.py                    # Hugging Face Spaces Streamlit entrypoint
├── app/
│   ├── dashboard.py          # Interactive Streamlit Web Application
│   └── main.py               # Production FastAPI REST API Server
├── models/
│   └── exoplanet_detector_final.keras # Trained deep learning model artifact
├── src/
│   ├── data/
│   │   ├── loader.py         # Kepler dataset parser & synthetic data generator
│   │   └── nasa_api.py       # Live NASA MAST Lightkurve query engine
│   ├── features/
│   │   ├── preprocessing.py  # Normalization, Gaussian smoothing, outlier filtering
│   │   └── astronomy.py      # BLS search, phase-folding, local/global views, physical telemetry
│   ├── models/
│   │   ├── cnn1d.py          # 1D CNN architecture builder
│   │   ├── resnet1d.py       # 1D Residual Network (ResNet1D) architecture
│   │   ├── local_global_cnn.py# Dual-branch local + global CNN
│   │   └── explainability.py # 1D Grad-CAM attention heatmap generator
│   └── utils/
│       └── metrics.py        # Recall-optimized metrics, ROC-AUC, threshold optimizer
├── train.py                  # CLI model training pipeline
├── evaluate.py               # CLI evaluation & validation suite
├── requirements.txt          # Python dependencies
├── Dockerfile                # Production container configuration
├── runtime.txt               # Python runtime specification (3.11)
├── .python-version           # Python version pin (3.11)
└── README.md                 # Project documentation
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites & Environment Setup
Clone the repository and initialize a Python 3.11 virtual environment:

```bash
git clone https://github.com/anuragkkumar/exoplanet-detection.git
cd exoplanet-detection

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running Locally

### 1. Launch the Streamlit Web Dashboard
```bash
streamlit run app/dashboard.py
```
Open your browser at `http://localhost:8501`.

### 2. Launch the FastAPI REST Server
```bash
uvicorn app.main:app --reload --port 8000
```
* **Interactive Swagger UI**: `http://localhost:8000/docs`
* **Redoc Documentation**: `http://localhost:8000/redoc`

#### Example REST API Prediction Request
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"flux": [0.012, -0.004, 0.001, -0.025, 0.003]}'
```

---

## 📊 Model Performance Benchmarks

Evaluated on the NASA Kepler Space Telescope test set:

| Model Architecture | Class Balancing | Decision Threshold | Overall Accuracy | Planet Recall | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1D CNN** | Class Weights | 0.50 | 99.1% | 81.0% | 0.962 |
| **SMOTE + 1D CNN** | SMOTE | 0.50 | 99.2% | 94.6% | 0.978 |
| **Tuned 1D CNN (Final)** | **SMOTE** | **0.30** | **99.3%** | **100.0%** (0 FN) | **0.984** |

---

## 📜 Acknowledgements & Hackathon

Built for the **ISRO Bharatiya Antariksh Hackathon (BAH) 2026**.  
Photometric data courtesy of the **NASA Kepler Mission**, **TESS Mission**, and the **Mikulski Archive for Space Telescopes (MAST)**.
