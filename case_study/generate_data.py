import numpy as np
import pandas as pd


def generate_nonstationary_series(fs, duration, frequency_segments, noise_std=0.01, trend_fn=None):
    """
    Generate smooth non-stationary time series with gradually changing frequency content.
    
    Parameters
    ----------
    frequency_segments : list of tuples
        Each tuple: (start_freq, end_freq, start_time, end_time, amplitude)
    """
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    x = np.zeros_like(t)
    
    for start_freq, end_freq, start_time, end_time, amplitude in frequency_segments:
        # Create time mask for this segment
        segment_mask = (t >= start_time) & (t <= end_time)
        segment_t = t[segment_mask] - start_time
        segment_duration = end_time - start_time
        
        # Smooth amplitude envelope (raised cosine taper)
        taper_length = min(segment_duration * 0.2, 0.5)  # 20% taper or max 0.5s
        envelope = np.ones_like(segment_t)
        
        # Apply tapers at beginning and end
        start_taper = segment_t < taper_length
        end_taper = segment_t > (segment_duration - taper_length)
        
        envelope[start_taper] = 0.5 * (1 - np.cos(np.pi * segment_t[start_taper] / taper_length))
        envelope[end_taper] = 0.5 * (1 + np.cos(np.pi * (segment_t[end_taper] - (segment_duration - taper_length)) / taper_length))
        
        # Smooth frequency transition (linear chirp)
        instantaneous_freq = start_freq + (end_freq - start_freq) * segment_t / segment_duration
        
        # Generate phase by integrating frequency
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / fs
        
        # Create windowed sinusoid
        segment_signal = amplitude * envelope * np.sin(phase)
        x[segment_mask] += segment_signal
    
    # Add trend if provided
    if trend_fn is not None:
        x += trend_fn(t)
    
    # Add noise
    x += np.random.normal(scale=noise_std, size=n_samples)
    
    return t, x

def generate_dataset(sampling_rate, duration):
    fs = sampling_rate
    
    # Define smooth frequency segments with overlapping tapered transitions
    exog1_segments = [
        (7, 3, 1, 9, 1.0),
        (7, 12, 2, 7, 1.8),
        (20, 12, 8, 10, 1.2),
        (25, 42, 6, 6.5, 0.9),        
        (40, 45, 2, 4, 3.1)
    ]
    
    exog2_segments = [
        (35, 22, 0, 6, 0.8),
        (4, 8, 3, 4, 3.3),
        (12, 15, 7.5, 10, 0.7),
        (25, 30, 5, 7, 2),
        (1, 6, 5, 7, 0.5)
    ]
    
    endog_segments = [
        (5, 7, 0, 5, 1.1),
        (21, 18, 3, 7, 5),
        (44, 40, 3, 4.5, 6),
        (1, 5, 2, 6, 4.1),
        (40, 36, 7.5, 9.9, 5.3),
        (11, 16, 7, 9, 2.7),
        (9, 6, 8, 9.5, 1.5),
    ]
    
    # Generate smooth trends
    trend_exog1 = lambda t: 0.01 * t * np.sin(0.5 * t)
    trend_exog2 = lambda t: 0.5 * np.exp(-0.1 * t) * np.cos(0.3 * t)
    trend_endog = lambda t: 0.02 * t**1.5 * np.sin(0.2 * t)
    
    # Generate the series
    time, exog1 = generate_nonstationary_series(fs, duration, exog1_segments, 
                                                      noise_std=0.05, trend_fn=trend_exog1)
    _, exog2 = generate_nonstationary_series(fs, duration, exog2_segments, 
                                                   noise_std=0.05, trend_fn=trend_exog2)
    _, endog = generate_nonstationary_series(fs, duration, endog_segments, 
                                                   noise_std=0.05, trend_fn=trend_endog)
    
    return exog1, exog2, endog, time