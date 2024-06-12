import os
import pickle

import pandas as pd
import numpy as np
import random as rd

# creates appropriate folders for data processing in the current directory.
def init_processing():
    """
    Creates the following file structure in the current working directory:\n
    .data/\n
        session/ - raw session data\n
        processed/ - processed session data\n
        chunks/ - processed chunks\n
        training/ - training data
    """
    data_path = "data"
    session_path = os.path.join(data_path, "session")
    processed_path = os.path.join(data_path, "processed")
    chunks_path = os.path.join(data_path, "chunks")
    training_path = os.path.join(data_path, "training")
    os.makedirs(session_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)
    os.makedirs(chunks_path, exist_ok=True)
    os.makedirs(training_path, exist_ok=True)

# creates chunk files from a labeled session file (processed or raw)
def prepare_chunk_data(filename, output_dir, start_identifiers=[], sep='\t'):
    # Define chunk end identifiers
    end_identifiers = [-x for x in start_identifiers]

    # Read the CSV file
    df = pd.read_csv(filename, sep=sep)

    # Variables to keep track of the current chunk and counters for each identifier
    current_chunk = []
    counters = {start: 0 for start in start_identifiers}

    # Iterate through the DataFrame
    for _, row in df.iterrows():
        last_col_value = row.iloc[-1]

        if last_col_value in start_identifiers:
            
            # Start a new chunk
            if current_chunk:
                current_chunk = []  # Reset the current chunk if it was not empty
            current_chunk = [row]

        elif last_col_value in end_identifiers and current_chunk:
            start_identifier = abs(last_col_value)
            if current_chunk[0].iloc[-1] == start_identifier:

                # End the chunk if the end identifier matches the start identifier
                current_chunk.append(row)

                # Save the chunk to a file
                chunk_df = pd.DataFrame(current_chunk)
                filename_prefix = f"{int(start_identifier)}_{counters[start_identifier]}"
                chunk_df.to_csv(os.path.join(output_dir, f'{filename_prefix}.csv'), header=False, index=False, sep=sep)
                counters[start_identifier] += 1
                current_chunk = []

        elif current_chunk:

            # If in a chunk, add the row to the current chunk
            current_chunk.append(row)

    return True

# processes the chunk files and creates training files
def process_chunks(chunk_dir, output_dir, func, sep='\t'):
    """
    Processes CSV files in the specified chunk directory using a given function,
    and saves the resulting data to separate files in the output directory.

    Parameters:
    chunk_dir (str): The directory containing the input CSV files.
    output_dir (str): The directory where the processed files will be saved.
    func (callable): A function that takes a file path as input and returns a list of dictionaries.
                     Each dictionary contains multiple DataFrames, with the keys representing data names.
    sep (str, optional): The separator to use when saving the CSV files. Defaults to '\t'.

    The `func` parameter:
    - The provided function should accept a single argument: the file path of a CSV file.
    - It should return a list of dictionaries. Each dictionary should have string keys and 
      pandas DataFrame values. For example:
      [
          {
              'timeseries': pd.DataFrame(...),
              'fftdata': pd.DataFrame(...),
              'other_data': pd.DataFrame(...),
              ...
          },
          ...
      ]

    Usage:
    - Place the input CSV files in the `chunk_dir`.
    - Ensure the `output_dir` exists or can be created.
    - Define a processing function `func` that returns the appropriate structure.
    - Call `process_chunks(chunk_dir, output_dir, func)` to process the files.

    Example:
    ```python
    def example_func(file_path):
        # Process the file and return a list of dictionaries with DataFrames
        return [
            {
                'timeseries': pd.DataFrame(...),
                'fftdata': pd.DataFrame(...),
                'other_data': pd.DataFrame(...),
            },
            ...
        ]

    process_chunks('/path/to/chunks', '/path/to/output', example_func)
    ```

    """
    for filename in os.listdir(chunk_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(chunk_dir, filename)
            chunk_result = func(file_path)
            
            # Save each chunk to a separate file
            for idx, data_dict in enumerate(chunk_result):
                name, ext = os.path.splitext(filename)
                
                for data_name, data_frame in data_dict.items():
                    processed_filename = f"{name}_{idx + 1}_{data_name}{ext}"
                    processed_file_path = os.path.join(output_dir, processed_filename)
                    os.makedirs(output_dir, exist_ok=True)

                    # Save the data to a new file
                    data_frame.to_csv(processed_file_path, index=False, header=False, sep=sep)


# creates a training data set from training files. TODO Make a full documentation for this so there's no confusion.
def create_training_data(input_dir, output_file, groups, func_dict, sep='\t'):

    """
    Creates training data from files in the input directory, applies given functions, and saves the result to a pickle file.
    
    Parameters:
    - input_dir (str): The directory containing the input data files.
    - output_file (str): The pickle output file.
    - groups (list)(int): A list of integer values that represent the present data labels.
    - func_dict (dict): A dictionary where keys are file type identifiers and values are functions to process those files.

    Returns:
    - a dictionary with 'groups' as keys, each containing a list of dictionaries with keys matching the keys in 'func_dict' and an additional 'label' key.

    IMPORTANT: Files in input_dir must be in form of idx1_idx2_idx3_filetype.csv, where:
    - idx1 is the label, 
    - idx2 is the chunk index (insignificant but must match across different 'filetype' files),
    - idx3 is the packer number of that chunk.

    """
    print(output_file)
    # Dictionary to hold the grouped data
    grouped_data = {group: [] for group in groups}
    
    # Iterate through files in the input directory
    for filename in os.listdir(input_dir):
        # Split the filename to extract idx1, idx2, idx3, and filetype
        parts = filename.split('_')
        if len(parts) < 4:
            continue  # Skip files that do not match the expected pattern
        
        idx1 = int(parts[0])  # Label
        idx2 = parts[1]  # Chunk index
        idx3 = parts[2]  # Packer number
        filetype = parts[3].split('.')[0]  # File type without the extension

        # Check if the label is in the specified groups
        if idx1 not in groups:
            continue

        # Read the file
        filepath = os.path.join(input_dir, filename)
        data = np.genfromtxt(filepath, delimiter=sep)
        
        # Process the data using the corresponding function
        if filetype in func_dict:
            processed_data = func_dict[filetype](data)
        else:
            continue  # Skip if there is no corresponding function for this file type
        
        # Prepare the data dictionary
        data_dict = {
            'label': idx1,
            filetype: processed_data
        }
        
        # Check if there's already an entry for this chunk
        existing_entry = next((entry for entry in grouped_data[idx1] if entry.get('chunk_index') == idx2 and entry.get('packer_number') == idx3), None)
        
        if existing_entry:
            # Update the existing entry with the new file type data
            existing_entry[filetype] = processed_data
        else:
            # Create a new entry
            new_entry = {
                'chunk_index': idx2,
                'packer_number': idx3,
                'label': idx1,
                filetype: processed_data
            }
            grouped_data[idx1].append(new_entry)

    # Save the grouped data to a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(grouped_data, f)

    return grouped_data
