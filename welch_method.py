import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

def compute_alpha_power(data, fs=200):
    f, Pxx = welch(data, fs=fs, nperseg=fs//2)  # Power Spectral Density (PSD)
    alpha_idx = np.where((f >= 8) & (f <= 12))  # Select 8-12 Hz range
    alpha_power = np.sum(Pxx[alpha_idx])  # Sum power in alpha range
    return alpha_power

def compute_alpha_power_time(data, fs=200, window_size = 600, step = 100, threshold = 3.25*(10**-5)):
    alpha_data, x_data = [], []
    for i in range(0, len(data)-window_size, step):
        alpha_power = compute_alpha_power(channel0[i:i+window_size], fs)
        alpha_data.append(alpha_power)
        x_data.append(i/fs)
        # if alpha_power > threshold:
        #     print("eyes closed at ", i/fs, "seconds")
        # else:
        #     print("eyes open at ", i/fs, "seconds")
    return np.array(alpha_data), np.array(x_data)

def get_ground_truth(time):
    """Returns ground truth based on time: True (alpha expected) or False (no alpha)."""
    if 0 <= time <= 30 or 60 <= time <= 90:  # Eyes closed (alpha present)
        return True
    elif 30 < time <= 60 or 90 < time <= 120:  # Eyes open (no alpha)
        return False
    return None  # Out of range


df = pd.read_csv('filtered_data.csv')
channel0 = df['Channel 0 Filtered'].values
channel1 = df['Channel 1 Filtered'].values

# create sliding window
fs = 200
window_size = int(fs*3) # 1 second window
step = fs//2 # 50% overlap
threshold = 3.25*(10**-5)

alpha_data, x_data = compute_alpha_power_time(channel0, window_size=window_size, step = step, threshold=threshold)


# Define threshold range (fine-tuned search)
thresholds = np.arange(0, 5e-5, 1e-6)  # Range: 0 to 5e-5 in steps of 1e-6
errors = []

for trial_threshold in thresholds:
    false_positives = 0
    false_negatives = 0

    for alpha, time in zip(alpha_data, x_data):
        true_label = get_ground_truth(time)  # Expected label
        predicted_label = alpha > trial_threshold  # Prediction using threshold

        if true_label is not None:  # Only evaluate within experiment range
            if predicted_label and not true_label:  # False Positive
                false_positives += 1
            if not predicted_label and true_label:  # False Negative
                false_negatives += 1

    # Total error count (FP + FN)
    total_error = false_positives + false_negatives
    errors.append((trial_threshold, total_error))

# Find the optimal threshold (minimizing total error)
optimal_threshold, min_error = min(errors, key=lambda x: x[1])
print(f"Optimal threshold: {optimal_threshold:.6e} (Min Error: {min_error})")

plt.figure(figsize=(10, 5))
plt.plot(x_data, alpha_data, label="Alpha Power", color="blue")
plt.axhline(optimal_threshold, color='r', linestyle='--', label="Optimal Threshold")
plt.xlabel('Time (s)')
plt.ylabel('Alpha Power')
plt.title('Alpha Power over Time')
plt.legend()
plt.show()
