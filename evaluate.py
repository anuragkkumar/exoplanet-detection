"""
Evaluation CLI Script for Exoplanet Detection System
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from src.data.loader import load_kepler_data, generate_synthetic_lightcurve
from src.features.preprocessing import preprocess_pipeline
from src.utils.metrics import calculate_evaluation_metrics


def evaluate_model(model_path="models/exoplanet_detector_final.keras", data_dir=".", threshold=0.3):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train a model first with train.py.")
        
    print(f"--> Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    test_file = os.path.join(data_dir, "exoTest.txt")
    if os.path.exists(test_file):
        X_test_raw, y_test = load_kepler_data(data_dir, "exoTest.txt")
    else:
        print("--> Test file exoTest.txt not found. Generating synthetic test set...")
        X_test_raw, y_test = [], []
        for i in range(100):
            has_p = (i < 10)
            flux, lbl, _ = generate_synthetic_lightcurve(has_planet=has_p)
            X_test_raw.append(flux)
            y_test.append(lbl)
        X_test_raw = np.array(X_test_raw)
        y_test = np.array(y_test)

    X_test_proc = preprocess_pipeline(X_test_raw, sigma=1.0)
    X_test_final = X_test_proc.reshape(X_test_proc.shape[0], X_test_proc.shape[1], 1)
    
    y_pred_prob = model.predict(X_test_final)
    metrics = calculate_evaluation_metrics(y_test, y_pred_prob, threshold=threshold)
    
    print("\n" + "="*50)
    print("           EVALUATION METRICS SUMMARY           ")
    print("="*50)
    print(f" Classification Threshold: {metrics['threshold']}")
    print(f" Accuracy:                 {metrics['accuracy'] * 100:.2f}%")
    print(f" Planet Recall:            {metrics['recall'] * 100:.2f}%")
    print(f" Precision:                {metrics['precision'] * 100:.2f}%")
    print(f" F1-Score:                 {metrics['f1_score']:.4f}")
    print(f" ROC-AUC Score:            {metrics['auc_score']:.4f}")
    print("="*50)
    print(f" Confusion Matrix (TN, FP, FN, TP):")
    print(f" [[{metrics['tn']}, {metrics['fp']}],")
    print(f"  [{metrics['fn']}, {metrics['tp']}]]")
    print("="*50)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Exoplanet Detection Model")
    parser.add_argument("--model_path", type=str, default="models/exoplanet_detector_final.keras", help="Path to saved model")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing exoTest.txt")
    parser.add_argument("--threshold", type=float, default=0.3, help="Decision threshold")
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.threshold)
