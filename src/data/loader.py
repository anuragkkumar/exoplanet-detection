import os
import pandas as pd
import numpy as np


def load_kepler_data(data_dir=".", filename="exoTrain.txt"):
    """
    Loads Kepler labeled time-series dataset from file.
    
    Parameters:
        data_dir (str): Directory containing the data file.
        filename (str): Name of the dataset file (e.g. exoTrain.txt or exoTest.txt).
        
    Returns:
        tuple: (flux matrix X of shape (N, T), binary labels y of shape (N,))
    """
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
    df = pd.read_csv(file_path)
    y = (df["LABEL"].values - 1).astype(int)  # Map 1 -> 0 (No planet), 2 -> 1 (Planet)
    flux = df.iloc[:, 1:].values
    return flux, y


def load_custom_lightcurve(file_input):
    """
    Parses a user-uploaded CSV or TXT light curve file into a 1D flux numpy array.
    """
    if isinstance(file_input, (str, os.PathLike)):
        df = pd.read_csv(file_input)
    else:
        df = pd.read_csv(file_input)

    # Handle 1D array or single-row dataset
    if df.shape[0] == 1:
        flux = df.iloc[0].values
    elif df.shape[1] == 1:
        flux = df.iloc[:, 0].values
    else:
        # Check if there is a header or TIME column
        non_num_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_num_cols) > 0:
            df = df.drop(columns=non_num_cols)
        # Use first flux column or first row
        flux = df.values.flatten()
        
    return flux.astype(np.float32)


def generate_synthetic_lightcurve(num_points=3197, has_planet=True, noise_level=0.5, period=600, depth=0.03):
    """
    Generates a realistic synthetic light curve time series for testing/demo.
    
    Parameters:
        num_points (int): Length of time series timesteps.
        has_planet (bool): Whether to inject a periodic transit dip.
        noise_level (float): Gaussian noise amplitude.
        period (int): Orbital period in timesteps.
        depth (float): Relative transit depth (fraction of flux drop).
        
    Returns:
        tuple: (flux array, binary label 0 or 1, true transit indices)
    """
    np.random.seed(42 if not has_planet else np.random.randint(1, 10000))
    time = np.arange(num_points)
    
    # Baseline stellar variability (low frequency sine wave + stellar spots)
    flux = 1.0 + 0.005 * np.sin(2 * np.pi * time / 1500)
    
    # Gaussian instrument noise
    flux += np.random.normal(0, noise_level * 0.01, size=num_points)
    
    transit_indices = []
    if has_planet:
        # Transit dips occurring periodically
        transit_width = 30
        for center in range(period // 2, num_points, period):
            start = max(0, center - transit_width // 2)
            end = min(num_points, center + transit_width // 2)
            # Trapezoidal transit shape
            dip_shape = np.ones(end - start)
            wing = max(1, (end - start) // 4)
            dip_shape[:wing] = np.linspace(1.0, 1.0 - depth, wing)
            dip_shape[wing:-wing] = 1.0 - depth
            dip_shape[-wing:] = np.linspace(1.0 - depth, 1.0, wing)
            
            flux[start:end] *= dip_shape
            transit_indices.extend(list(range(start, end)))
            
    return flux, int(has_planet), transit_indices
