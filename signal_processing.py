import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Normalize signals to [-1, 1] range
def normalize_signal(data):
    """Normalize signal to [-1, 1] range."""
    return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1
# 1. Filtering
def apply_bandpass_filter(data, lowcut, highcut, fs=200, order=4):
    """Apply a bandpass filter to the signal."""
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

# notch filter
def notch_filter(data, freq=60, fs=200, quality_factor=30):
    b, a = signal.iirnotch(freq / (fs / 2), quality_factor)
    return signal.filtfilt(b, a, data)
# 2. Moving Average (Smoothing)
def moving_average(data, window_size):
    """Apply moving average smoothing."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 3. Power Spectrum
def compute_power_spectrum(data, fs=200):
    """Compute the power spectrum of the signal."""
    frequencies, psd = signal.welch(data, fs=fs)
    return frequencies, psd

# 4. Root Mean Square (RMS)
def compute_rms(data, window_size):
    """Compute RMS value over sliding windows."""
    return np.sqrt(np.convolve(data**2, np.ones(window_size)/window_size, mode='valid'))

# 5. Envelope Detection
def compute_envelope(data):
    """Compute signal envelope using Hilbert transform."""
    analytic_signal = signal.hilbert(data)
    envelope = np.abs(analytic_signal)
    return envelope

