import pandas as pd
import os
from brainflow import DataFilter, WindowOperations
import sys
import numpy as np


def process_chunk(file_path, N, X, skip_rows=250) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load the CSV file
    df = pd.read_csv(file_path, header=None)

    # List to store the results
    result = []

    # Iterate through the DataFrame in steps of N+X
    i = skip_rows
    while i < len(df):
        # Skip N rows
        end_index = i + N

        # Ensure no out of bounds
        start_index = end_index - X
        if end_index > len(df):
            break

        # Take the last X rows
        chunk = df.iloc[start_index:end_index]

        # Process the data
        timeseries_data = chunk.iloc[-N:]
        timeseries_data = timeseries_data[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        chunk_np = chunk.to_numpy().transpose()

        data = [[], ]
        for channel in range(1, 9):
            fft_data = DataFilter.get_psd_welch(chunk_np[channel], X, X // 2, skip_rows, WindowOperations.BLACKMAN_HARRIS)
            lim = min(32, len(fft_data[0]))
            values = fft_data[0][0:lim].tolist()
            data[0] = fft_data[1][0:lim].tolist()
            data.append(values)

        fft = np.array(data)
        fft = pd.DataFrame(fft)

        result.append((timeseries_data, fft))

        # Move the index forward
        i = end_index

    return result



# Define N and X
N = 32  # packet size
X = 128   # Number of rows to take for fft 

# Directory containing the chunk files
chunk_dir = 'data/chunks_processed/2/'
output_dir = os.path.join(chunk_dir, 'training/')

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each chunk file
for filename in os.listdir(chunk_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(chunk_dir, filename)
        chunk_result = process_chunk(file_path, N, X)
        
        # Save each chunk to a separate file
        for idx, (timeseries_data, fft_data) in enumerate(chunk_result):
            name, ext = os.path.splitext(filename)
            processed_filename_timeseries = f"{name}_{idx + 1}_timedata{ext}"
            processed_filename_fftdata = f"{name}_{idx + 1}_fftdata{ext}"
            processed_file_path_timeseries = os.path.join(output_dir, processed_filename_timeseries)
            processed_file_path_fftdata = os.path.join(output_dir, processed_filename_fftdata)

            # Save the chunk to a new file
            timeseries_data.to_csv(processed_file_path_timeseries, index=False, header=False)
            fft_data.to_csv(processed_file_path_fftdata, index=False, header=False)
            # fft_data.tofile(processed_file_path_fftdata, sep=',')
        
        print(f'Processed and saved chunks from {filename}')
