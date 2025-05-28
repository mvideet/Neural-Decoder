import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# Read the data
df = pd.read_csv('extracted_data.csv')
channel0 = df['EXG Channel 0'].values
channel1 = df['EXG Channel 1'].values

# Normalize signals
def normalize_signal(data):
    return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

channel0_norm = normalize_signal(channel0)
channel1_norm = normalize_signal(channel1)

# Parameters for spectrogram
fs = 200  # Sampling frequency (Hz)
nperseg = 256  # Length of each segment
noverlap = nperseg // 2  # Number of points to overlap
nfft = 512  # Length of FFT

def create_spectrogram(data, channel_name):
    # Calculate spectrogram using STFT
    frequencies, times, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, 
                                               noverlap=noverlap, nfft=nfft,
                                               window='hann', scaling='spectrum')
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Add small constant to avoid log(0)
    return frequencies, times, Sxx_db

# Create spectrograms for both channels
f0, t0, Sxx0 = create_spectrogram(channel0_norm, 'Channel 0')
f1, t1, Sxx1 = create_spectrogram(channel1_norm, 'Channel 1')

# Plotting
plt.figure(figsize=(15, 10))

# Channel 0 Spectrogram
plt.subplot(2, 1, 1)
plt.pcolormesh(t0, f0, Sxx0, shading='gouraud', cmap='viridis')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylabel('Frequency [Hz]')
plt.title('Spectrogram - Channel 0')
plt.ylim(0, fs//2)  # Limit to Nyquist frequency

# Channel 1 Spectrogram
plt.subplot(2, 1, 2)
plt.pcolormesh(t1, f1, Sxx1, shading='gouraud', cmap='viridis')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram - Channel 1')
plt.ylim(0, fs//2)  # Limit to Nyquist frequency

plt.tight_layout()
plt.savefig('spectrograms.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some frequency analysis statistics
for i, Sxx in enumerate([Sxx0, Sxx1]):
    print(f"\nChannel {i} Frequency Analysis:")
    # Find dominant frequencies
    mean_power = np.mean(Sxx, axis=1)
    dominant_freq_idx = np.argsort(mean_power)[-3:]  # Top 3 frequencies
    print("Top 3 dominant frequencies:")
    for idx in reversed(dominant_freq_idx):
        print(f"{f0[idx]:.1f} Hz (Power: {mean_power[idx]:.1f} dB)")
