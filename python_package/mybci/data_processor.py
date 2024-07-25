import os
import time
import pickle
import shutil

import numpy as np
import pandas as pd

from collections import deque
from brainflow import BoardShim, BoardIds, BrainFlowInputParams


def save_to_file(data: pd.DataFrame, path, name, sep):
    file = os.path.join(path, name)
    data.to_csv(path_or_buf=file, sep=sep, header=None, index=False)



# creates a training data set from training files. TODO Make a full documentation for this so there's no confusion.
def create_training_data(data_list, groups, func_dict={}, sep='\t'):

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
    - idx3 is the packet number of that chunk.
    """
    # Dictionary to hold the grouped data
    grouped_data = {group: [] for group in groups}
    
    for data_entry in data_list:
        idx1 = int(data_entry[0])
        idx2 = data_entry[1]
        idx3 = data_entry[2]
        feature = data_entry[3]

        # Check if the label is in the specified groups
        if idx1 not in groups:
            continue

        # Read the file
        data = data_entry[4]

        # Process the data using the corresponding function
        #TODO implement this option, won't do for now because no use cases.
        processed_data = data

        # Check if there's already an entry for this chunk
        existing_entry = next((entry for entry in grouped_data[idx1] if entry.get('chunk_index') == idx2 and entry.get('packet_number') == idx3), None)
        
        if existing_entry:
            # Update the existing entry with the new file type data
            existing_entry[feature] = processed_data
        else:
            # Create a new entry
            new_entry = {
                'label': idx1,
                'chunk_index': idx2,
                'packet_number': idx3,
                feature: processed_data
            }
            grouped_data[idx1].append(new_entry)

    return grouped_data









class PacketReader:
    
    def __init__(self, data_source, packet_size, buffer_size, is_live=False, sep='\t'):
        self.data_source = data_source
        self.is_live = is_live
        self.packet_size = packet_size
        self.buffer = deque(maxlen=buffer_size)
        self.sep=sep

        if self.is_live:
            from brainflow.board_shim import BoardShim
            if not isinstance(self.data_source, BoardShim):
                raise ValueError("For live mode, data_source must be an instance of BoardShim.")
            self.board = self.data_source
            self.get_data_packet = self._get_live_data
            self.name = "live"
        else:
            if not isinstance(self.data_source, str):
                raise ValueError("For non-live mode, data_source must be a file path.")
            self.filepath = self.data_source
            self.data_iter = self._file_data_generator()
            self.get_data_packet = self._get_file_data
            self.name = os.path.splitext(os.path.basename(data_source))[0]

    def _file_data_generator(self):
        for packet in pd.read_csv(self.filepath, chunksize=self.packet_size, sep=self.sep):
            yield packet.values

    def _get_live_data(self):
        if self.board.get_board_data_count() >= self.packet_size:
            data = self.board.get_board_data(self.packet_size).transpose()
            return data
        else:
            return np.array([]) # Not enough data available

    def _get_file_data(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            return np.array([])  # No more data in the file



class ChunkHandler:
    def __init__(self, start_identifiers=[], sep='\t'):
        self.start_identifiers = start_identifiers
        self.end_identifiers = [-x for x in start_identifiers]
        self.sep = sep
        self.current_chunk = []
        self.counters = {start: 0 for start in start_identifiers}
        

    def process_data_packet(self, data_packet):
        # Convert the data packet to a DataFrame
        df = pd.DataFrame(data_packet)
        output = {'complete': False}
        # Iterate through the DataFrame
        for _, row in df.iterrows():
            last_col_value = row.iloc[-1]

            if last_col_value in self.start_identifiers:
                # Start a new chunk
                if self.current_chunk:
                    self.current_chunk = []  # Reset the current chunk if it was not empty
                self.current_chunk = [row]

            elif last_col_value in self.end_identifiers and self.current_chunk:
                start_identifier = abs(last_col_value)
                if self.current_chunk[0].iloc[-1] == start_identifier:
                    # End the chunk if the end identifier matches the start identifier
                    self.current_chunk.append(row)

                    # Save the chunk
                    # name = f"{int(start_identifier)}_{self.counters[start_identifier]}"
                    output['data'] = pd.DataFrame(self.current_chunk)
                    output['label'] = int(start_identifier)
                    output['counter'] = self.counters[start_identifier]
                    output['complete'] = True
                    self.counters[start_identifier] += 1
                    self.current_chunk = []

            elif self.current_chunk:
                # If in a chunk, add the row to the current chunk
                self.current_chunk.append(row)
                
        return output

    def flush_current_chunk(self):
        output = {}
        
        if self.current_chunk:
            start_identifier = abs(self.current_chunk[0].iloc[-1])
            output['data'] = pd.DataFrame(self.current_chunk)
            output['label'] = int(start_identifier)
            output['counter'] = self.counters[start_identifier]
            self.counters[start_identifier] += 1
            self.current_chunk = []
            
        return output


from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class Config:
    """
    Configuration class for setting parameters with descriptions and default values.

    Attributes:
        name (str): Name of the configuration.
        action_markers (List[int]): List of action markers.
        buffer_size (int): Size of the buffer, from which the data will be filtered.
        packet_size (int): Size of the packets.
        feature_size (int): Size of the feature.
        chunk_offset (int): Offset for data chunking.
        sampling_rate (int): Sampling rate of the data.
        filter_function (Callable[[int], bool]): Function used for filtering data.
        feature_function (Callable[[int], int]): Function used for feature extraction.
        session_folders (List[str]): List of session folder paths.
        excluded_session_files (List[str]): List of files to exclude from sessions.
        dataset_directory (str): Directory path for datasets.
        keep_helper_files (bool): Whether to keep helper files after processing.
        live_board (BoardIDs): Identifier for the live board.
        device_port (str): Port for the device connection.
        prediction_function (Callable[[int], int]): Function used for making predictions.
    """
    name: str = 'Data Processing'
    action_markers: List[int] = field(default_factory= lambda: [1, 2])
    buffer_size: int = 128
    packet_size: int = 32
    feature_size: int = 128
    chunk_offset: int = 196
    sampling_rate: int = 250
    filter_function: Callable = None
    feature_function: Callable = None
    session_folders: List[str] = field(default_factory= lambda: [])
    excluded_session_files: List[str] = field(default_factory= lambda: [])
    channel_column_ids: List[int] = field(default_factory= lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    use_live_board: bool = False
    live_board: BoardIds = BoardIds.CYTON_BOARD.value
    electrode_config: list = field(default_factory= lambda: [])
    device_port: str = ''
    prediction_function: Callable = None
    live_pipeline = None

    separator: str = '\t'
    dataset_directory: str = 'data/datasets/'
    helper_files_location: str = 'data/temp/'
    keep_helper_files: bool = False


class DataProcessor:

    def __init__(self, config:Config):
        self.config = config
        # self.params = dict(config)
        self.chunk_handlers = {}
        self.readers = {}

        if self.config.use_live_board:
            self.readers['live'] = self._create_live_packet_reader()
            self.filtered_buffer = deque(maxlen=self.config.feature_size) # maybe it should be buffer_size?
            self._update_func = self._update_live if not self.config.live_pipeline else self._update_live_pipeline
            self.update = lambda: self._update(self.readers['live'])
        else:
            files = []
            for folder in self.config.session_folders:
                for file in os.listdir(folder):
                    skip = 0
                    for excluded in self.config.excluded_session_files:
                        if excluded in file:
                            skip = 1
                            break
                    if not skip:
                        files.append(os.path.join(folder, file))

            for file in files:
                filename = os.path.splitext(os.path.basename(file))[0]
                self.chunk_handlers[filename] = ChunkHandler(self.config.action_markers, self.config.separator)
                self.readers[filename] = PacketReader(file, self.config.packet_size, self.config.buffer_size, False, self.config.separator)
                self.current_dataset = []

            self._update_func = self._update_file

        self.output_path = os.path.join(self.config.dataset_directory, self.config.name)
        os.makedirs(self.output_path, exist_ok=True)

    def process(self):
        for filename, reader in self.readers.items():
            print(filename)
            while self._update(reader):
                pass

    def _update(self, reader) -> bool:
        packet = reader.get_data_packet()
        reader.buffer.extend(packet)
        status = self._update_func(packet, reader.buffer, reader.name)
        return status

    def _create_live_packet_reader(self):
        source = self._open_board()
        return PacketReader(source, self.config.packet_size, self.config.buffer_size, True, self.config.separator)

    def _open_board(self):
        params = BrainFlowInputParams()
        params.serial_port = params['board_device']
        board = BoardShim(self.config.live_board, params)
        board.prepare_session()
        for conf in self.config.electrode_config:
            print("Sending", conf)
            board.config_board(conf)
        board.start_stream()
        return board

    def _update_live_pipeline(self, packet: np.ndarray):
        """NOT IMPLEMENTED YET, FOR NOW USE 'session.py' SCRIPT."""
        pass

    def _update_live(self, packet: np.ndarray, buffer: deque, name: str):
        if not packet.size:
            # handle timeouts here
            return 1
        
        if len(buffer) < self.config.feature_size:
            return 1
        
        # filtered_packet = self.config.filter_function(np.array(buffer)[-self.config.filter_size:], self.config)
        self.filtered_buffer.extend(packet)
        if len(self.filtered_buffer) < self.config.feature_size:
            return 1

        data = self.config.feature_function(np.array(self.filtered_buffer)[-self.config.feature_size:], self.config)
        self.config.prediction_function(data, self.config)


    def _update_file(self, packet: np.ndarray, buffer: deque, name: str):
        chunk_handler = self.chunk_handlers[name]
        if not packet.size:
            return self.create_training_sets(name)

        # filtered_packet = self.config.filter_function(np.array(buffer)[-self.config.packet_size:, ], self.config)
        filtered_packet = self.config.filter_function(np.array(buffer), self.config)[-self.config.packet_size:]
        # filtered_packet = self.config.filter_function(np.array(buffer), self.config)
        chunk = chunk_handler.process_data_packet(filtered_packet)
        if not chunk['complete']:
            return 1

        data = self._iterate_chunk(chunk, self.config.feature_function)

        for idx, ml_packet in enumerate(data):
            for data_name, ml_packet_data in ml_packet['data'].items():
                name = f"{ml_packet['label']}_{ml_packet['counter']}_{idx}_{data_name}.csv"
                # save_to_file(ml_packet_data, path, name, self.config.separator)
                self.current_dataset.append([ml_packet['label'], ml_packet['counter'], idx, data_name, ml_packet_data])

        return 1

    def _iterate_chunk(self, chunk: dict, func):
        output_data = []
        chunk_data = chunk['data']
        label = chunk['label']
        i = self.config.chunk_offset
        
        while i < len(chunk_data):
            i = i + self.config.packet_size
            end_index = i
            start_index = end_index - self.config.feature_size

            if end_index > len(chunk_data):
                break

            data = chunk_data.iloc[start_index:end_index, :]

            output = {'data': func(data, self.config), 'label': label, 'counter': chunk['counter']}
            output_data.append(output)

        return output_data

    def create_training_sets(self, filename):
        data = create_training_data(self.current_dataset, self.config.action_markers, {}, self.config.separator)
        output_file = os.path.join(self.output_path, f"dataset_{filename}.pickle")
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        self.current_dataset = []
        return 0
