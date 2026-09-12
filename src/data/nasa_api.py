"""
NASA API Module using Lightkurve
Queries official NASA Kepler, K2, and TESS archives for star light curves.
"""

import numpy as np


def fetch_nasa_lightcurve(target_id, mission="Kepler", target_points=3197):
    """
    Downloads and cleans light curve data from MAST archive using Lightkurve.
    
    Parameters:
        target_id (str or int): Target identifier (e.g. "KIC 10593626" or "10593626" or "TIC 25155310").
        mission (str): Mission name ("Kepler", "TESS", "K2").
        target_points (int): Resampled timestep length for model compatibility.
        
    Returns:
        dict: {
            "target_id": str,
            "mission": str,
            "raw_flux": np.ndarray,
            "time": np.ndarray,
            "resampled_flux": np.ndarray,
            "success": bool,
            "message": str
        }
    """
    target_str = str(target_id).strip()
    if not target_str.upper().startswith(("KIC", "TIC", "EPIC")):
        prefix = "TIC" if mission.upper() == "TESS" else "KIC"
        target_str = f"{prefix} {target_str}"
        
    try:
        import lightkurve as lk

        search_result = lk.search_lightcurve(target_str, mission=mission)
        if len(search_result) == 0:
            return {
                "target_id": target_str,
                "mission": mission,
                "success": False,
                "message": f"No light curves found on MAST for target {target_str} ({mission})."
            }
            
        # Download first available light curve product
        lc = search_result[0].download()
        lc = lc.remove_nans().remove_outliers()
        
        time = lc.time.value
        flux = lc.flux.value
        
        # Interpolate / resample to fixed timesteps for neural net input shape
        time_interp = np.linspace(time[0], time[-1], target_points)
        flux_interp = np.interp(time_interp, time, flux)
        
        return {
            "target_id": target_str,
            "mission": mission,
            "raw_flux": np.array(flux, dtype=np.float32),
            "time": np.array(time, dtype=np.float32),
            "resampled_flux": np.array(flux_interp, dtype=np.float32),
            "success": True,
            "message": f"Successfully retrieved light curve for {target_str} ({len(time)} points)."
        }
        
    except ImportError:
        # Fallback for testing when lightkurve is not available
        from .loader import generate_synthetic_lightcurve
        synth_flux, _, _ = generate_synthetic_lightcurve(num_points=target_points, has_planet=True)
        return {
            "target_id": target_str,
            "mission": mission,
            "raw_flux": synth_flux,
            "time": np.arange(target_points, dtype=np.float32),
            "resampled_flux": synth_flux,
            "success": True,
            "message": f"Lightkurve module not installed. Generated synthetic light curve for demo of {target_str}."
        }
    except Exception as e:
        return {
            "target_id": target_str,
            "mission": mission,
            "success": False,
            "message": f"Error querying NASA MAST archive: {str(e)}"
        }
