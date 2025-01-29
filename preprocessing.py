import csv

# Input and output file paths
input_file = './raw-earlobeL-earlobeR-O1-P3.txt'
output_file = './extracted_data.csv'

# Columns to extract
print("HEllo")
columns_to_extract = ["EXG Channel 0", "EXG Channel 1", "Timestamp"]

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

# Extract the required data
data = []
for line in lines:
    if line.startswith("Sample Index") or line.startswith("%") or not line.strip():
        continue
    row = line.strip().split(",")
    extracted_row = [row[column_indices[col]] for col in columns_to_extract]
    data.append(extracted_row)

# Write the extracted data to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns_to_extract)  # Write the header
    writer.writerows(data)  # Write the data

print(f"Extracted data saved to {output_file}")