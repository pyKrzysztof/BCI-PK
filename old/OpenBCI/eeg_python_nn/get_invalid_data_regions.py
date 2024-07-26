import csv
import pandas as pd
import os

# Read the CSV file
input_file = 'dane/processed_constant_detrend.csv'
output_dir = 'dane/chunks_processed_constant_detrend/zero_chunks/'
data = pd.read_csv(input_file, sep="\t")

# Initialize variables
chunks = []
current_chunk = []
inside_chunk = False
open_tag = None

# Iterate through each row
for index, row in data.iterrows():
    last_column_value = row.iloc[-1]
    
    # Check for the start of a chunk
    if last_column_value in [1, 2] and not inside_chunk:
        inside_chunk = True
        open_tag = last_column_value
    elif last_column_value in [-1, -2] and inside_chunk and open_tag == -last_column_value:
        inside_chunk = False
        open_tag = None
    
    # Add row to the current chunk if not inside an enclosed chunk
    if not inside_chunk:
        current_chunk.append(row)
    else:
        # If inside a chunk, save current chunk if not empty and reset
        if current_chunk:
            chunks.append(pd.DataFrame(current_chunk))
            current_chunk = []

# Save any remaining chunk if not empty
if current_chunk:
    chunks.append(pd.DataFrame(current_chunk))

os.makedirs(output_dir, exist_ok=True)

# Save each chunk to a separate CSV file
for i, chunk in enumerate(chunks):
    chunk.to_csv(os.path.join(output_dir, f'ZeroData{i + 1}.csv'), index=False, header=False)

print(f'Total {len(chunks)} chunks found and saved.')
