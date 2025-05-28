import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

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

scaler = StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

num_classes = 2
y_train_categorical=keras.utils.to_categorical(y_train, 2)
y_test_categorical=keras.utils.to_categorical(y_test, 2)

model =keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(features.shape[1],)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
            
              loss='categorical_crossentropy',
              metrics=['accuracy'])
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train_categorical, epochs=20, batch_size=16, validation_data=(X_test_scaled, y_test_categorical), callbacks=[early_stopping])
test_loss, test_acc = model.evaluate(X_test_scaled, y_test_categorical)
print(f"Test accuracy: {test_acc}")
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred_classes)
y_true = np.argmax(y_test_categorical, axis=1)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred_classes))
confusion_matrix = confusion_matrix(y_true, y_pred_classes)
print(confusion_matrix)
