import numpy as np
from scipy.ndimage import gaussian_filter1d


def normalize_flux(X):
    """
    Performs per-star Z-score standard normalization across time-series flux.
    (flux - mean) / std for each row.
    """
    X_clean = np.nan_to_num(X, nan=0.0)
    if X_clean.ndim == 1:
        mean = np.mean(X_clean)
        std = np.std(X_clean) + 1e-8
        return (X_clean - mean) / std
    else:
        mean = np.mean(X_clean, axis=1).reshape(-1, 1)
        std = np.std(X_clean, axis=1).reshape(-1, 1) + 1e-8
        return (X_clean - mean) / std


def smooth_flux(X, sigma=1.0):
    """
    Applies Gaussian 1D filter smoothing to reduce sharp high-frequency random noise.
    """
    X_clean = np.nan_to_num(X, nan=0.0)
    if X_clean.ndim == 1:
        return gaussian_filter1d(X_clean, sigma=sigma)
    else:
        return np.array([gaussian_filter1d(x, sigma=sigma) for x in X_clean])


def preprocess_pipeline(X, sigma=1.0):
    """
    Complete preprocessing pipeline: NaN replacement -> Z-score normalization -> Gaussian smoothing.
    
    Parameters:
        X (np.ndarray): 1D array of shape (T,) or 2D array of shape (N, T).
        sigma (float): Gaussian filter standard deviation.
        
    Returns:
        np.ndarray: Cleaned and normalized flux array.
    """
    norm_x = normalize_flux(X)
    if sigma > 0:
        return smooth_flux(norm_x, sigma=sigma)
    return norm_x
