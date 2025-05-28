from signal_processing import *
# Read the data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('extracted_data.csv')
channel0 = df['EXG Channel 0'].values
channel1 = df['EXG Channel 1'].values
time = df['Timestamp'].values

#filtering parameters
fs = 200
low, high = 3, 30
notch_freq = 60
quality_factor = 30


channel0_norm = normalize_signal(channel0)
channel1_norm = normalize_signal(channel1)

channel0_bandpass = apply_bandpass_filter(channel0_norm, low, high)
channel1_bandpass = apply_bandpass_filter(channel1_norm, low, high)

channel0_filtered = notch_filter(channel0_bandpass, notch_freq, fs, quality_factor)
channel1_filtered = notch_filter(channel1_bandpass, notch_freq, fs, quality_factor)


filtered_data = pd.DataFrame({
    'Channel 0 Filtered': channel0_filtered,
    'Channel 1 Filtered': channel1_filtered
})

# Save the DataFrame to a CSV file
filtered_data.to_csv('filtered_data.csv', index=False)

