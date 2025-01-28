import numpy as np
import librosa
signal_ch1 = data['Channel 1'].values
signal_ch2 = data['Channel 2'].values

# Define parameters
sr = 22050  # Sampling rate (adjust as needed)
n_fft = 2048
hop_length = 512

# Generate spectrograms
spectrogram_ch1 = librosa.stft(signal_ch1, n_fft=n_fft, hop_length=hop_length)
spectrogram_db_ch1 = librosa.amplitude_to_db(np.abs(spectrogram_ch1), ref=np.max)

spectrogram_ch2 = librosa.stft(signal_ch2, n_fft=n_fft, hop_length=hop_length)
spectrogram_db_ch2 = librosa.amplitude_to_db(np.abs(spectrogram_ch2), ref=np.max)
