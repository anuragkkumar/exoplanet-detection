"""
Astronomical Signal Processing & Transit Analysis Module
Implements Box-Fitting Least Squares (BLS), Phase Folding, Local/Global View extraction, and Planet Parameter estimation.
"""

import numpy as np


def phase_fold(time, flux, period, t0=0.0):
    """
    Folds light curve time-series over a specified orbital period.
    
    Parameters:
        time (np.ndarray): Time array.
        flux (np.ndarray): Flux array.
        period (float): Orbital period (same units as time).
        t0 (float): Epoch center time of transit.
        
    Returns:
        tuple: (sorted_phase, sorted_flux) where phase is centered between [-0.5, 0.5].
    """
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]


def find_bls_period(time, flux, min_period=50.0, max_period=1000.0, num_periods=500):
    """
    Estimates primary transit periodicity and epoch t0 using Box-fitting Least Squares (BLS) heuristic.
    
    Returns:
        dict: {
            "best_period": float,
            "best_t0": float,
            "transit_depth": float,
            "transit_duration": float
        }
    """
    try:
        from astropy.timeseries import BoxLeastSquares

        model = BoxLeastSquares(time, flux)
        periods = np.linspace(min_period, max_period, num_periods)
        # Search over reasonable transit durations (e.g. 5 to 50 timesteps)
        duration = np.linspace(5, 50, 10)
        results = model.power(periods, duration)
        
        best_idx = np.argmax(results.power)
        best_period = float(results.period[best_idx])
        best_t0 = float(results.transit_time[best_idx])
        transit_depth = float(results.depth[best_idx])
        transit_duration = float(results.duration[best_idx])
        
        return {
            "best_period": best_period,
            "best_t0": best_t0,
            "transit_depth": transit_depth,
            "transit_duration": transit_duration
        }
    except Exception:
        # Fallback period detection based on minimum flux dip spacing
        min_idx = np.argmin(flux)
        t0 = float(time[min_idx]) if time is not None else float(min_idx)
        # Find secondary minimum far enough from primary
        dist = np.abs(np.arange(len(flux)) - min_idx)
        flux_masked = flux.copy()
        flux_masked[dist < 50] = np.max(flux)
        sec_min_idx = np.argmin(flux_masked)
        period = float(abs(sec_min_idx - min_idx))
        if period < 10:
            period = 600.0
            
        depth = float(np.max(flux) - np.min(flux))
        return {
            "best_period": period,
            "best_t0": t0,
            "transit_depth": depth,
            "transit_duration": 30.0
        }


def extract_local_global_views(flux, local_window_size=201):
    """
    Generates NASA/Google AI dual-view representations:
    1. Global View: Full normalized light curve (e.g. 3197 timesteps).
    2. Local View: Zoomed-in window centered at the primary transit dip (e.g. 201 timesteps).
    
    Parameters:
        flux (np.ndarray): 1D light curve flux.
        local_window_size (int): Size of zoomed local transit window (must be odd).
        
    Returns:
        tuple: (global_view, local_view)
    """
    flux_clean = np.nan_to_num(flux, nan=0.0)
    N = len(flux_clean)
    
    # Locate primary transit minimum
    min_idx = np.argmin(flux_clean)
    
    half_w = local_window_size // 2
    start = min_idx - half_w
    end = min_idx + half_w + 1
    
    # Pad if near edges
    if start < 0:
        local_view = np.pad(flux_clean[0:end], (-start, 0), mode='edge')
    elif end > N:
        local_view = np.pad(flux_clean[start:N], (0, end - N), mode='edge')
    else:
        local_view = flux_clean[start:end]
        
    return flux_clean, local_view


def estimate_planet_radius(transit_depth, stellar_radius_solar=1.0):
    """
    Estimates candidate planet radius in Earth Radii (R_earth) from relative transit dip depth.
    
    Formula:
        (R_p / R_*)^2 = depth  =>  R_p = R_* * sqrt(depth)
        R_sun = 109.2 * R_earth
        
    Parameters:
        transit_depth (float): Relative fractional flux drop (e.g., 0.01 = 1%).
        stellar_radius_solar (float): Radius of host star in Solar Radii (R_sun). Default 1.0.
        
    Returns:
        dict: {
            "planet_radius_earth": float,
            "planet_radius_jupiter": float,
            "depth_percent": float,
            "classification": str
        }
    """
    depth = max(0.0, float(transit_depth))
    r_p_over_r_star = np.sqrt(depth)
    
    # 1 R_sun = 109.076 R_earth = 9.731 R_jupiter
    r_earth = r_p_over_r_star * stellar_radius_solar * 109.076
    r_jupiter = r_earth / 11.209
    
    if r_earth < 1.25:
        category = "Earth-sized"
    elif r_earth < 2.0:
        category = "Super-Earth"
    elif r_earth < 6.0:
        category = "Neptune-like"
    else:
        category = "Gas Giant / Jupiter-like"
        
    return {
        "planet_radius_earth": round(r_earth, 2),
        "planet_radius_jupiter": round(r_jupiter, 2),
        "depth_percent": round(depth * 100, 3),
        "classification": category
    }
