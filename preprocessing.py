import csv
import numpy as np
from scipy.io.wavfile import write

# Input and output file paths
input_file = './raw-earlobeL-earlobeR-O1-P3.txt'
output_file = './extracted_data.csv'
sonic_file = './sonic_data.csv'
sonic_wav_file = './sonic_data.wav'

# Columns to extract
columns_to_extract = ["EXG Channel 0", "EXG Channel 1", "Timestamp", "Timestamp (Formatted)"]
sonic_columns = ["EXG Channel 0", "EXG Channel 1"]

# Sampling rate for the WAV file (e.g., 200 Hz for EEG data or higher for audio)
sampling_rate = 200

# Read and process the input file
with open(input_file, 'r') as txtfile:
    lines = txtfile.readlines()

# Find the header row
header_line = None
for line in lines:
    if line.startswith("Sample Index"):
        header_line = line.strip().split(", ")
        break

if not header_line:
    raise ValueError("Header row not found in the file.")

# Get the indices of the desired columns
column_indices = {col: header_line.index(col) for col in columns_to_extract}
sonic_indices = [header_line.index(col) for col in sonic_columns]

# Extract the required data
data = []
sonic_data = []
for line in lines:
    if line.startswith("Sample Index") or line.startswith("%") or not line.strip():
        continue
    row = line.strip().split(",")
    # Extract full data for CSV
    extracted_row = [row[column_indices[col]] for col in columns_to_extract]
    data.append(extracted_row)
    # Extract only EXG data for Sonic Visualizer
    sonic_row = [float(row[i]) for i in sonic_indices]
    sonic_data.append(sonic_row)

# Write the extracted data to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns_to_extract)  # Write the header
    writer.writerows(data)  # Write the data

# Write the Sonic Visualizer formatted data to a CSV file
with open(sonic_file, 'w', newline='') as csvfile2:
    writer2 = csv.writer(csvfile2)
    writer2.writerows(sonic_data)

# Normalize and save as a WAV file for Sonic Visualizer
sonic_data = np.array(sonic_data)
max_amplitude = np.max(np.abs(sonic_data))
sonic_data_normalized = (sonic_data / max_amplitude * 32767).astype(np.int16)
write(sonic_wav_file, sampling_rate, sonic_data_normalized)

print(f"Extracted data saved to {output_file}")
print(f"Sonic Visualizer data saved to {sonic_file}")
print(f"Sonic Visualizer WAV file saved to {sonic_wav_file}")
