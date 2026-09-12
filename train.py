"""
Training CLI Script for Exoplanet Detection System
Usage:
    python train.py --model_type 1d_cnn --epochs 10 --use_smote
"""

import os
import argparse
import numpy as np
from imblearn.over_sampling import SMOTE

from src.data.loader import load_kepler_data, generate_synthetic_lightcurve
from src.features.preprocessing import preprocess_pipeline
from src.models.cnn1d import build_1d_cnn
from src.models.resnet1d import build_1d_resnet
from src.utils.metrics import calculate_evaluation_metrics, optimize_decision_threshold


def train_pipeline(data_dir=".", model_type="1d_cnn", epochs=10, batch_size=32, use_smote=True, smooth_sigma=1.0, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(data_dir, "exoTrain.txt")
    test_file = os.path.join(data_dir, "exoTest.txt")
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        print(f"--> Loading Kepler dataset from {data_dir}...")
        X_train_raw, y_train = load_kepler_data(data_dir, "exoTrain.txt")
        X_test_raw, y_test = load_kepler_data(data_dir, "exoTest.txt")
    else:
        print("--> Dataset files (exoTrain.txt/exoTest.txt) not found. Generating synthetic dataset for demo training...")
        train_samples = []
        y_train = []
        for i in range(500):
            has_p = (i < 30)
            flux, lbl, _ = generate_synthetic_lightcurve(has_planet=has_p)
            train_samples.append(flux)
            y_train.append(lbl)
            
        test_samples = []
        y_test = []
        for i in range(100):
            has_p = (i < 10)
            flux, lbl, _ = generate_synthetic_lightcurve(has_planet=has_p)
            test_samples.append(flux)
            y_test.append(lbl)
            
        X_train_raw = np.array(train_samples)
        y_train = np.array(y_train)
        X_test_raw = np.array(test_samples)
        y_test = np.array(y_test)

    print(f"--> Preprocessing flux (Normalizing & Gaussian Smoothing sigma={smooth_sigma})...")
    X_train_proc = preprocess_pipeline(X_train_raw, sigma=smooth_sigma)
    X_test_proc = preprocess_pipeline(X_test_raw, sigma=smooth_sigma)
    
    if use_smote:
        print("--> Applying SMOTE to balance class distribution...")
        sm = SMOTE(random_state=42)
        X_train_flat = X_train_proc.reshape(X_train_proc.shape[0], -1)
        X_resampled, y_train_resampled = sm.fit_resample(X_train_flat, y_train)
        X_train_final = X_resampled.reshape(X_resampled.shape[0], X_resampled.shape[1], 1)
        y_train_final = y_train_resampled
        print(f"    Class counts after SMOTE: {np.bincount(y_train_final)}")
    else:
        X_train_final = X_train_proc.reshape(X_train_proc.shape[0], X_train_proc.shape[1], 1)
        y_train_final = y_train
        
    X_test_final = X_test_proc.reshape(X_test_proc.shape[0], X_test_proc.shape[1], 1)

    print(f"--> Building model architecture: {model_type}...")
    if model_type == "1d_resnet":
        model = build_1d_resnet(input_shape=(X_train_final.shape[1], 1))
    else:
        model = build_1d_cnn(input_shape=(X_train_final.shape[1], 1))

    model.summary()

    print(f"--> Training for {epochs} epochs...")
    history = model.fit(
        X_train_final, y_train_final,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    print("--> Evaluating model on test set...")
    y_pred_prob = model.predict(X_test_final)
    
    best_thresh = optimize_decision_threshold(y_test, y_pred_prob, target_recall=1.0)
    print(f"--> Optimal Threshold for 100% Recall: {best_thresh:.3f}")
    
    metrics = calculate_evaluation_metrics(y_test, y_pred_prob, threshold=best_thresh)
    print("--> Evaluation Results:")
    print(f"    Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"    Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"    Precision: {metrics['precision'] * 100:.2f}%")
    print(f"    ROC-AUC:   {metrics['auc_score']:.4f}")

    save_path = os.path.join(output_dir, "exoplanet_detector_final.keras")
    model.save(save_path)
    print(f"--> Model saved successfully to {save_path}!")
    
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Exoplanet Detection Neural Network")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing dataset files")
    parser.add_argument("--model_type", type=str, default="1d_cnn", choices=["1d_cnn", "1d_resnet"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--use_smote", action="store_true", default=True, help="Use SMOTE balancing")
    parser.add_argument("--smooth_sigma", type=float, default=1.0, help="Gaussian smoothing sigma")
    parser.add_argument("--output_dir", type=str, default="models", help="Saved model directory")
    
    args = parser.parse_args()
    train_pipeline(
        data_dir=args.data_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_smote=args.use_smote,
        smooth_sigma=args.smooth_sigma,
        output_dir=args.output_dir
    )
