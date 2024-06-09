import os
import time
import pickle
import numpy as np
import pandas as pd

from lib import DataProcessor, prepare_chunk_data
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations, FilterTypes

""" File structure:

.data/
    session/ - raw session data - input directory
    processed/ - processed session data - first stage
    chunks/ - processed chunks - second stage
    training/ - training data - output directory
    process_functions/ - processing scripts directory - configuration
"""

data_path = "data"
session_path = os.path.join(data_path, "session")
processed_path = os.path.join(data_path, "processed")
chunks_path = os.path.join(data_path, "chunks")
training_path = os.path.join(data_path, "training")

os.makedirs(processed_path, exist_ok=True)
os.makedirs(chunks_path, exist_ok=True)
os.makedirs(training_path, exist_ok=True)


#######################################################################
# temporary
N = 32
X = 128
actions = [1, 2]
def _process(data, is_full):
    timeseries_data = np.array(data.transpose(), order="C")
    for channel in [1, 2, 3, 4, 5, 6, 7, 8]:
        DataFilter.detrend(timeseries_data[channel], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(timeseries_data[channel], 250, 4.0, 45.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 48.0, 52.0, 2,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 58.0, 62.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

    timeseries_data = timeseries_data.transpose()[-32:]
    return timeseries_data

def process_chunk(file_path, N, X, skip_rows=250, sep='\t') -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load the CSV file
    df = pd.read_csv(file_path, header=None, sep=sep)

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
#######################################################################


def process_chunks(chunk_dir, output_dir, func, sep='\t'):
    for filename in os.listdir(chunk_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(chunk_dir, filename)
            chunk_result = func(file_path, N, X)
            
            # Save each chunk to a separate file
            for idx, (timeseries_data, fft_data) in enumerate(chunk_result):
                name, ext = os.path.splitext(filename)
                processed_filename_timeseries = f"{name}_{idx + 1}_timedata{ext}"
                processed_filename_fftdata = f"{name}_{idx + 1}_fftdata{ext}"
                processed_file_path_timeseries = os.path.join(output_dir, processed_filename_timeseries)
                processed_file_path_fftdata = os.path.join(output_dir, processed_filename_fftdata)

                # Save the chunk to a new file
                timeseries_data.to_csv(processed_file_path_timeseries, index=False, header=False, sep=sep)
                fft_data.to_csv(processed_file_path_fftdata, index=False, header=False, sep=sep)


def create_training_data(input_dir, output_dir, groups=[], sep='\t'):
    # Create a dictionary to store training data for each group
    training_data_dict = {str(i): [] for i in groups}

    # Load training data
    for filename in os.listdir(input_dir):
        # Skip fft data, it will be loaded manually
        if not filename.endswith("timedata.csv"):
            continue

        idx1, idx2, idx3, _ = filename.split("_") # label_chunk_packet_timedata.csv
        fft_filename = f"{idx1}_{idx2}_{idx3}_fftdata.csv"
        timedata = np.genfromtxt(os.path.join(input_dir, filename), delimiter=sep)
        fftdata = np.genfromtxt(os.path.join(input_dir, fft_filename), delimiter=sep)

        # Reject data with crazy accel (not implemented yet)

        # Remove accel channels
        timedata = timedata[:, :8]

        # Remove frequency scale
        fftdata = fftdata[1:, :]

        # Get label
        label = int(idx1)

        # Append data to the appropriate group in the dictionary
        group_key = idx1[0]
        if group_key in training_data_dict:
            training_data_dict[group_key].append((timedata, fftdata, label))
        else:
            print(f"Warning: Filename {filename} does not match any expected group")

    # Save the collected data to separate output files
    for group_key, data_list in training_data_dict.items():
        output_filename = os.path.join(output_dir, f"{group_key}.pickle")
        with open(output_filename, 'wb') as f:
            pickle.dump(data_list, f)
        print(f"Saved group {group_key} training data to {output_filename}")



def process_session_data(session_data, stages=None):
    dir_name = session_data[:-4] + "/"
    session_file = os.path.join(session_path, session_data)
    processed_file = os.path.join(processed_path, session_data)
    chunk_dir = os.path.join(chunks_path, dir_name)

    os.makedirs(chunk_dir, exist_ok=True)
    training_dir = os.path.join(training_path, dir_name)
    os.makedirs(training_dir, exist_ok=True)
    training_data_dir = os.path.join(training_dir, "data/")
    os.makedirs(training_data_dir, exist_ok=True)

    if stages is None or 0 in stages: 
        processor = DataProcessor(session_file, N, _process, X, processed_file, sep='\t')
        processor.process_data_sources()
    if stages is None or 1 in stages: 
        prepare_chunk_data(processed_file, chunk_dir, actions)
    if stages is None or 2 in stages: 
        process_chunks(chunk_dir, training_data_dir, process_chunk, sep='\t')
    if stages is None or 3 in stages: 
        create_training_data(training_data_dir, training_dir, actions, sep='\t')


for session_data in os.listdir(session_path):
    if not session_data.endswith('.csv'):
        continue
    process_session_data(session_data, stages=[])

