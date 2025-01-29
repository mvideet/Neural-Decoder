import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Read the CSV file
df = pd.read_csv('extracted_data.csv')

# Create a figure with a specific size
plt.figure(figsize=(12, 6))

# Plot both channels
plt.plot(df['Timestamp'], df['EXG Channel 0'], label='EXG Channel 0', color='blue')
plt.plot(df['Timestamp'], df['EXG Channel 1'], label='EXG Channel 1', color='red')

# Customize the plot
plt.title('EXG Channel Data over Time', fontsize=14)
plt.xlabel('Timestamp')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('exg_channels_plot.png')
plt.close()
