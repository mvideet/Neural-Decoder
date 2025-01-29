# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.signal import welch

# def compute_alpha_power(data, fs=200):
#     f, Pxx = welch(data, fs=fs, nperseg=fs//2)  # Power Spectral Density (PSD)
#     alpha_idx = np.where((f >= 8) & (f <= 12))  # Select 8-12 Hz range
#     alpha_power = np.sum(Pxx[alpha_idx])  # Sum power in alpha range
#     return alpha_power

# def compute_alpha_power_time(data, fs=200, window_size = 600, step = 100, threshold = 3.25*(10**-5)):
#     alpha_data, x_data = [], []
#     for i in range(0, len(data)-window_size, step):
#         alpha_power = compute_alpha_power(channel0[i:i+window_size], fs)
#         alpha_data.append(alpha_power)
#         x_data.append(i/fs)
#         # if alpha_power > threshold:
#         #     print("eyes closed at ", i/fs, "seconds")
#         # else:
#         #     print("eyes open at ", i/fs, "seconds")
#     return np.array(alpha_data), np.array(x_data)

# def get_ground_truth(time):
#     """Returns ground truth based on time: True (alpha expected) or False (no alpha)."""
#     if 0 <= time <= 30 or 60 <= time <= 90:  # Eyes closed (alpha present)
#         return False
#     elif 30 < time <= 60 or 90 < time <= 120:  # Eyes open (no alpha)
#         return True
#     return None  # Out of range


# df = pd.read_csv('filtered_data.csv')
# channel0 = df['Channel 0 Filtered'].values
# channel1 = df['Channel 1 Filtered'].values

# # create sliding window
# fs = 200
# window_size = int(fs*1)# 1 second window
# step = fs//2 # 50% overlap
# threshold = 3.25*(10**-5)

# alpha_data, x_data = compute_alpha_power_time(channel0, window_size=window_size, step = step, threshold=threshold)

#___________________________________________________________

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.decomposition import PCA
# Load EEG Data
df = pd.read_csv('filtered_data.csv')
channel0 = df['Channel 0 Filtered'].values
fs = 200  # Sampling frequency

# Define Alpha Power Calculation
def compute_alpha_power(data, fs=200):
    f, Pxx = welch(data, fs=fs, nperseg=fs//2)
    alpha_idx = np.where((f >= 8) & (f <= 12))
    return np.sum(Pxx[alpha_idx])  

# Feature Extraction Methods
def compute_relative_alpha_power(data, fs=200):
    f, Pxx = welch(data, fs=fs, nperseg=fs//2)
    alpha_idx = np.where((f >= 8) & (f <= 12))
    total_power = np.sum(Pxx)
    return np.sum(Pxx[alpha_idx]) / total_power if total_power > 0 else 0

def compute_alpha_peak_frequency(data, fs=200):
    f, Pxx = welch(data, fs=fs, nperseg=fs//2)
    alpha_idx = np.where((f >= 8) & (f <= 12))
    return f[alpha_idx][np.argmax(Pxx[alpha_idx])] if len(alpha_idx[0]) > 0 else 0

def compute_rms_alpha(data):
    return np.sqrt(np.mean(np.square(data)))

def compute_variance(data):
    return np.var(data)

def hjorth_mobility(data):
    first_derivative = np.diff(data)
    return np.std(first_derivative) / np.std(data) if np.std(data) > 0 else 0

def hjorth_complexity(data):
    first_derivative = np.diff(data)
    second_derivative = np.diff(first_derivative)
    return hjorth_mobility(second_derivative) / hjorth_mobility(first_derivative) if hjorth_mobility(first_derivative) > 0 else 0

# Compute Features Over Time
def compute_features_over_time(data, fs=200, window_size=600, step=100):
    feature_data, time_data = [], []
    for i in range(0, len(data) - window_size, step):
        segment = data[i:i + window_size]
        features = [
            compute_alpha_power(segment, fs),
            compute_relative_alpha_power(segment, fs),
            compute_alpha_peak_frequency(segment, fs),
            compute_rms_alpha(segment),
            compute_variance(segment),
            hjorth_mobility(segment),
            hjorth_complexity(segment)
        ]
        feature_data.append(features)
        time_data.append(i / fs)  # Convert sample index to time (seconds)
    return np.array(feature_data), np.array(time_data)

# Sliding Window Parameters
window_size = int(fs * 3)  # 3-second window
step = fs // 2  # 50% overlap

# Extract Features
features, x_data = compute_features_over_time(channel0, window_size=window_size, step=step)
print(features[:,0].reshape(-1,1).shape)
# Define Ground Truth for Alpha Wave Detection
def get_ground_truth(time):
    """Returns ground truth based on time: True (alpha expected) or False (no alpha)."""
    if 0 <= time <= 30 or 60 <= time <= 90:  # Eyes closed (high alpha)
        return 1  # Alpha present
    elif 30 < time <= 60 or 90 < time <= 120:  # Eyes open (low alpha)
        return 0  # Alpha absent
    return None  

# Create Labels for Machine Learning
labels = np.array([get_ground_truth(t) for t in x_data])
valid_idx = labels != None  # Filter out None values
features = features[valid_idx]
labels = labels[valid_idx]

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(features[:,:], labels, test_size=0.2, random_state=42)

# Train SVM Model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Predict on Test Data
y_pred = svm_model.predict(X_test)

# Compute Model Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print Model Performance
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:\n", conf_matrix)

# Plot Alpha Power Over Time
plt.figure(figsize=(10, 5))
plt.plot(x_data, features[:, 0], label="Alpha Power", color="blue")
plt.xlabel('Time (s)')
plt.ylabel('Alpha Power')
plt.title('Alpha Power over Time')
plt.legend()
#plt.show()
## **Apply PCA for Dimensionality Reduction (7D → 3D)**
pca = PCA(n_components=3)
features_pca = pca.fit_transform(features)

# Scatter Plot in 3D (Visualizing Alpha vs. Non-Alpha States)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Color by Alpha Presence (Red = No Alpha, Green = Alpha Detected)
colors = ['red' if label == 0 else 'green' for label in labels]

ax.scatter(features_pca[:, 0], features_pca[:, 1], features_pca[:, 2], c=colors, s=40, edgecolors='k', alpha=0.8)

# Labels & Title
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3D PCA Visualization of EEG Features')
#plt.show()

#__________________
#2d visualization
# **Apply PCA for Dimensionality Reduction (7D → 2D)**
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

# Scatter Plot in 2D (Visualizing Alpha vs. Non-Alpha States)
plt.figure(figsize=(10, 7))

# Color by Alpha Presence (Red = No Alpha, Green = Alpha Detected)
colors = ['red' if label == 0 else 'green' for label in labels]

plt.scatter(features_pca[:, 0], features_pca[:, 1], c=colors, s=40, edgecolors='k', alpha=0.8)

# Labels & Title
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('2D PCA Visualization of EEG Features')
plt.grid(True)
#plt.show()