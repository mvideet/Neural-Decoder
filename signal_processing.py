import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('extracted_data.csv')
channel0 = df['EXG Channel 0'].values
channel1 = df['EXG Channel 1'].values
time = df['Timestamp'].values

# Normalize signals to [-1, 1] range
def normalize_signal(data):
    """Normalize signal to [-1, 1] range."""
    return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

channel0_norm = normalize_signal(channel0)
channel1_norm = normalize_signal(channel1)

# 1. Filtering
def apply_bandpass_filter(data, lowcut, highcut, fs=200):
    """Apply a bandpass filter to the signal."""
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
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

# Process both channels
def process_channel(data, channel_name):
    filtered = apply_bandpass_filter(data, 4, 30)
    smoothed = moving_average(data, 20)  # Reduced window size for better visualization
    freq, psd = compute_power_spectrum(data)
    rms = compute_rms(data, 20)
    envelope = compute_envelope(data)
    return filtered, smoothed, freq, psd, rms, envelope

# Process normalized signals
ch0_filtered, ch0_smoothed, ch0_freq, ch0_psd, ch0_rms, ch0_envelope = process_channel(channel0_norm, 'Channel 0')
ch1_filtered, ch1_smoothed, ch1_freq, ch1_psd, ch1_rms, ch1_envelope = process_channel(channel1_norm, 'Channel 1')

# Create visualization
fig = plt.figure(figsize=(15, 20))

# 1. Original Signals
plt.subplot(5, 2, 1)
plt.plot(channel0_norm[:1000], label='Channel 0', alpha=0.8)
plt.title('Original Normalized Signal - Channel 0')
plt.grid(True)
plt.legend()

plt.subplot(5, 2, 2)
plt.plot(channel1_norm[:1000], label='Channel 1', alpha=0.8, color='orange')
plt.title('Original Normalized Signal - Channel 1')
plt.grid(True)
plt.legend()

# 2. Filtered Signals
plt.subplot(5, 2, 3)
plt.plot(ch0_filtered[:1000], label='Filtered', alpha=0.8)
plt.title('Bandpass Filtered (4-30 Hz) - Channel 0')
plt.grid(True)
plt.legend()

plt.subplot(5, 2, 4)
plt.plot(ch1_filtered[:1000], label='Filtered', alpha=0.8, color='orange')
plt.title('Bandpass Filtered (4-30 Hz) - Channel 1')
plt.grid(True)
plt.legend()

# 3. Smoothed Signals
plt.subplot(5, 2, 5)
plt.plot(ch0_smoothed[:1000], label='Smoothed', alpha=0.8)
plt.title('Smoothed Signal - Channel 0')
plt.grid(True)
plt.legend()

plt.subplot(5, 2, 6)
plt.plot(ch1_smoothed[:1000], label='Smoothed', alpha=0.8, color='orange')
plt.title('Smoothed Signal - Channel 1')
plt.grid(True)
plt.legend()

# 4. Power Spectrum
plt.subplot(5, 2, 7)
plt.semilogy(ch0_freq, ch0_psd, label='PSD', alpha=0.8)
plt.title('Power Spectrum - Channel 0')
plt.xlabel('Frequency [Hz]')
plt.grid(True)
plt.legend()

plt.subplot(5, 2, 8)
plt.semilogy(ch1_freq, ch1_psd, label='PSD', alpha=0.8, color='orange')
plt.title('Power Spectrum - Channel 1')
plt.xlabel('Frequency [Hz]')
plt.grid(True)
plt.legend()

# 5. Signal with Envelope
plt.subplot(5, 2, 9)
plt.plot(channel0_norm[:1000], label='Signal', alpha=0.5)
plt.plot(ch0_envelope[:1000], label='Envelope', color='red', alpha=0.8)
plt.title('Signal Envelope - Channel 0')
plt.grid(True)
plt.legend()

plt.subplot(5, 2, 10)
plt.plot(channel1_norm[:1000], label='Signal', alpha=0.5, color='orange')
plt.plot(ch1_envelope[:1000], label='Envelope', color='red', alpha=0.8)
plt.title('Signal Envelope - Channel 1')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('normalized_signal_processing.png', dpi=300, bbox_inches='tight')
plt.close()

# Print statistics for normalized signals
print("\nNormalized Signal Statistics:")
print("\nChannel 0:")
print(f"Mean: {np.mean(channel0_norm):.3f}")
print(f"Standard deviation: {np.std(channel0_norm):.3f}")
print(f"Min: {np.min(channel0_norm):.3f}")
print(f"Max: {np.max(channel0_norm):.3f}")

print("\nChannel 1:")
print(f"Mean: {np.mean(channel1_norm):.3f}")
print(f"Standard deviation: {np.std(channel1_norm):.3f}")
print(f"Min: {np.min(channel1_norm):.3f}")
print(f"Max: {np.max(channel1_norm):.3f}")
