import os
import time
import pickle
import shutil

import numpy as np
import pandas as pd

from collections import deque
from brainflow import BoardShim


def save_to_file(data: pd.DataFrame, path, name, sep):
    file = os.path.join(path, name)
    data.to_csv(path_or_buf=file, sep=sep, header=None, index=False)



# creates a training data set from training files. TODO Make a full documentation for this so there's no confusion.
def create_training_data(input_dir, groups, func_dict={}, sep='\t'):

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
    
    # Iterate through files in the input directory
    for filename in os.listdir(input_dir):
        # Split the filename to extract idx1, idx2, idx3, and filetype
        parts = filename.split('_')
        if len(parts) < 4:
            continue  # Skip files that do not match the expected pattern
        
        idx1 = int(parts[0])  # Label
        idx2 = parts[1]  # Chunk index
        idx3 = parts[2]  # Packet number
        filetype = parts[3].split('.')[0]  # File type without the extension

        # Check if the label is in the specified groups
        if idx1 not in groups:
            continue
        

        # Read the file
        filepath = os.path.join(input_dir, filename)
        data = np.genfromtxt(filepath, delimiter=sep)
        
        # print(filename)
        # print(data.shape)

        # Process the data using the corresponding function
        if filetype in func_dict:
            processed_data = func_dict[filetype](data)
        else:
            processed_data = data

        # Check if there's already an entry for this chunk
        existing_entry = next((entry for entry in grouped_data[idx1] if entry.get('chunk_index') == idx2 and entry.get('packet_number') == idx3), None)
        
        if existing_entry:
            # Update the existing entry with the new file type data
            existing_entry[filetype] = processed_data
        else:
            # Create a new entry
            new_entry = {
                'label': idx1,
                'chunk_index': idx2,
                'packet_number': idx3,
                filetype: processed_data
            }
            grouped_data[idx1].append(new_entry)

    # print(grouped_data)

    return grouped_data









class PacketReader:
    
    def __init__(self, data_source, packet_size, is_live=False, sep='\t'):
        self.data_source = data_source
        self.is_live = is_live
        self.packet_size = packet_size
        self.sep=sep

        if self.is_live:
            from brainflow.board_shim import BoardShim
            if not isinstance(self.data_source, BoardShim):
                raise ValueError("For live mode, data_source must be an instance of BoardShim.")
            self.board = self.data_source
            self.get_data_packet = self._get_live_data
        else:
            if not isinstance(self.data_source, str):
                raise ValueError("For non-live mode, data_source must be a file path.")
            self.filepath = self.data_source
            self.data_iter = self._file_data_generator()
            self.get_data_packet = self._get_file_data

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



class NewDataProcessor:
    
    def __init__(self, config):
        self.params = dict(config)
        self.buffer = deque(maxlen=self.params['buffer_size'])
        self.reader = self._create_packet_reader()
        self.chunk_handlers = {name: ChunkHandler(self.params['action_markers'], self.params['sep']) for name in self.params['filter_func'].keys()}

        if self.params['use_board_device']:
            self._update_func = self._update_live if not self.params['live_pipeline'] else self._update_live_pipeline
        else:
            self._update_func = self._update_file

        if self.params['save_chunks']:
            self.params['output_path_chunks'] = self.params['output_path_chunks'] + self.params['name'] + "/"
            os.makedirs(self.params['output_path_chunks'], exist_ok=True)
        
        self.params['output_path_training_data'] = self.params['output_path_training_data'] + self.params['name'] + "/"
        os.makedirs(self.params['output_path_training_data'], exist_ok=True)
        os.makedirs(self.params['output_path_training_dataset'], exist_ok=True)



    def update(self) -> bool:
        packet = self.reader.get_data_packet()
        self.buffer.extend(packet) # this will ignore empty arrays and extend with actual packet rows.
        
        # if status returns 0 then it's over.
        status = self._update_func(packet)
        
        return status
        
    
    def _create_packet_reader(self):
        if self.params['use_board_device']:
            board = self._open_board()
            return PacketReader(board, self.params['packet_size'], True, self.params['sep'])
        
        return PacketReader(self.params['session_file'], self.params['packet_size'], False, self.params['sep'])
    
    def _open_board(self):
        return None

    def _update_live_pipeline(self, packet: np.ndarray):
        """NOT IMPLEMENTED YET, FOR NOW USE 'session.py' SCRIPT."""
        # process the pipeline
        # NOT IMPLEMENTED YET, FOR NOW USE 
        pass

    def _update_live(self, packet: np.ndarray):
        # check for timeout
        pass

    def _iterate_chunk(self, chunk: dict[str:int, str:int, str:pd.DataFrame], func):
        output_data = []
        chunk_data = chunk['data']
        label = chunk['label']
        i = self.params['ml_prepare_chunk_start_offset']
        while i < len(chunk_data):
            i = i + self.params['packet_size']
            end_index = i
            start_index = end_index - self.params['ml_prepare_size']

            # Ensure no out of bounds
            if end_index > len(chunk_data):
                break

            data = chunk_data.iloc[start_index:end_index, self.params['channel_column_ids'] + self.params['ml_prepare_extra_columns']]

            output = {'data': func(data, self.params), 'label': label, 'counter':chunk['counter']}
            output_data.append(output)


        return output_data

    def _update_file(self, packet: np.ndarray):
        if not packet.size: # file is empty, processing ended - create training sets if configured to do so.
            if self.params['save_training_dataset']:
                return self.create_training_sets()
            return 0
        
        for filter in self.params['filter_func'].keys():
            # print(filter)
            filtered_packet = self.params['filter_func'][filter](np.array(self.buffer)[-self.params['filter_size']:], self.params)
            chunk = self.chunk_handlers[filter].process_data_packet(filtered_packet) # output dict keys: label, counter, data, complete
            if not chunk['complete']:
                continue

            # save chunk
            if self.params['save_chunks']:
                path = os.path.join(self.params['output_path_chunks'], filter+'/')
                os.makedirs(path, exist_ok=True)
                name = f"{chunk['label']}_{chunk['counter']}.csv"
                save_to_file(chunk['data'], path, name, self.params['sep'])
                # print("Saved full chunk:", name)

            # ml prep
            for func in self.params['ml_prepare_func'].keys():
                # print(func)
                data = self._iterate_chunk(chunk, self.params['ml_prepare_func'][func])
                # print("chunk ml processed")
                # save ml data
                path = os.path.join(self.params['output_path_training_data'], filter+'/', func+'/')
                os.makedirs(path, exist_ok=True)
                
                for idx, ml_packet in enumerate(data):
                    # print(ml_packet)
                    for data_name, ml_packet_data in ml_packet['data'].items():
                        name = f"{ml_packet['label']}_{ml_packet['counter']}_{idx}_{data_name}.csv"
                        save_to_file(ml_packet_data, path, name, self.params['sep'])

        return 1


    def create_training_sets(self):
        for filter in self.params['filter_func'].keys():
            for func in self.params['ml_prepare_func'].keys():
                input_dir = os.path.join(self.params['output_path_training_data'], filter+'/', func+'/')
                data = create_training_data(input_dir, self.params['action_markers'], {}, self.params['sep'])
                
                output_file = os.path.join(self.params['output_path_training_dataset'], f"{self.params['name']}_{filter}_{func}.pickle")
                with open(output_file, 'wb') as f:
                    pickle.dump(data, f)

        if not self.params['keep_seperate_training_data']:
            for filter in self.params['filter_func'].keys():
                shutil.rmtree(self.params['output_path_training_data'][:-1])

        return 0