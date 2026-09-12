"""
FastAPI REST API Server for Exoplanet Detection System
Exposes model inference, 1D Grad-CAM explainability, and live NASA MAST data integration endpoints.
"""

import os
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

import tensorflow as tf

from src.features.preprocessing import preprocess_pipeline
from src.features.astronomy import find_bls_period, estimate_planet_radius, phase_fold
from src.models.explainability import compute_gradcam1d
from src.models.cnn1d import build_1d_cnn
from src.data.nasa_api import fetch_nasa_lightcurve
from src.data.loader import generate_synthetic_lightcurve

app = FastAPI(
    title="Exoplanet Transit Detection API",
    description="ISRO BAH 2026 AI Exoplanet Classification & Transit Analysis API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model Container
MODEL_PATH = "models/exoplanet_detector_final.keras"
_model = None


def get_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            try:
                _model = tf.keras.models.load_model(MODEL_PATH)
            except Exception:
                _model = build_1d_cnn()
        else:
            # Build and compile dummy/default model if saved model missing
            _model = build_1d_cnn()
    return _model


class LightCurveRequest(BaseModel):
    flux: List[float] = Field(..., description="Array of time-series stellar flux measurements")
    threshold: Optional[float] = Field(0.3, description="Classification decision threshold (default 0.3 for 100% recall)")
    stellar_radius_solar: Optional[float] = Field(1.0, description="Stellar radius in Solar Radii for planet radius estimation")


@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "Exoplanet Transit Detection System (ISRO BAH 2026)",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health_check():
    model_loaded = os.path.exists(MODEL_PATH)
    return {
        "status": "healthy",
        "model_file_exists": model_loaded,
        "model_path": MODEL_PATH,
        "api_version": "1.0.0"
    }


@app.post("/predict")
def predict_exoplanet(req: LightCurveRequest):
    if len(req.flux) == 0:
        raise HTTPException(status_code=400, detail="Flux array cannot be empty.")
        
    flux_arr = np.array(req.flux, dtype=np.float32)
    flux_proc = preprocess_pipeline(flux_arr, sigma=1.0)
    
    # Resample or pad to 3197 timesteps if necessary
    if len(flux_proc) != 3197:
        flux_proc = np.interp(np.linspace(0, 1, 3197), np.linspace(0, 1, len(flux_proc)), flux_proc)
        
    input_tensor = flux_proc.reshape(1, 3197, 1)
    
    model = get_model()
    prob = float(model.predict(input_tensor, verbose=0)[0, 0])
    is_planet = prob >= req.threshold
    
    bls_result = find_bls_period(np.arange(len(flux_arr)), flux_arr)
    radius_est = estimate_planet_radius(bls_result["transit_depth"], stellar_radius_solar=req.stellar_radius_solar)
    
    return {
        "is_exoplanet": bool(is_planet),
        "exoplanet_probability": round(prob, 4),
        "threshold_used": req.threshold,
        "classification": "Confirmed Exoplanet Candidate" if is_planet else "No Exoplanet Detected",
        "periodicity_analysis": bls_result,
        "planet_radius_estimation": radius_est
    }


@app.post("/explain")
def explain_prediction(req: LightCurveRequest):
    if len(req.flux) == 0:
        raise HTTPException(status_code=400, detail="Flux array cannot be empty.")
        
    flux_arr = np.array(req.flux, dtype=np.float32)
    flux_proc = preprocess_pipeline(flux_arr, sigma=1.0)
    
    if len(flux_proc) != 3197:
        flux_proc = np.interp(np.linspace(0, 1, 3197), np.linspace(0, 1, len(flux_proc)), flux_proc)
        
    model = get_model()
    prob, gradcam_heatmap = compute_gradcam1d(model, flux_proc)
    
    return {
        "exoplanet_probability": round(prob, 4),
        "gradcam_importance_heatmap": gradcam_heatmap.tolist(),
        "processed_flux": flux_proc.tolist()
    }


@app.get("/nasa/{target_id}")
def get_nasa_target(target_id: str, mission: str = Query("Kepler")):
    result = fetch_nasa_lightcurve(target_id, mission=mission)
    if not result["success"]:
        raise HTTPException(status_code=444 if "No light curves" in result["message"] else 500, detail=result["message"])
        
    model = get_model()
    flux_proc = preprocess_pipeline(result["resampled_flux"], sigma=1.0)
    input_tensor = flux_proc.reshape(1, 3197, 1)
    
    prob = float(model.predict(input_tensor, verbose=0)[0, 0])
    is_planet = prob >= 0.3
    
    return {
        "target_id": result["target_id"],
        "mission": result["mission"],
        "message": result["message"],
        "exoplanet_probability": round(prob, 4),
        "is_exoplanet": bool(is_planet),
        "classification": "Confirmed Exoplanet Candidate" if is_planet else "No Exoplanet Detected",
        "time": result["time"].tolist()[:500],  # Return truncated array for fast JSON payload
        "resampled_flux": flux_proc.tolist()
    }
