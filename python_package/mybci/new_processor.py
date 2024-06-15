import time

import numpy as np
import pandas as pd

from collections import deque
from brainflow import BoardShim



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
        output = {}
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
        self.params = config
        self.buffer = deque(maxlen=self.params['buffer_size'])
        self.reader = self._create_packet_reader()
        self.chunk_handlers = {name: ChunkHandler(self.params['action_markers'], self.params['sep']) for name in self.params['filter_func'].keys()}

        if self.params['use_board_device']:
            self._update_func = self._update_live if not self.params['live_pipeline'] else self._update_live_pipeline
        else:
            self._update_func = self._update_file


    def update(self) -> bool:
        packet = self.reader.get_data_packet()
        self.buffer.extend(packet) # this will ignore empty arrays and extend with actual packet rows.
        
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
        # process the pipeline
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

            end_index = i + self.params['packet_size']
            start_index = end_index - self.params['ml_prepare_size']

            # Ensure no out of bounds
            if end_index > len(chunk_data):
                break

            data = chunk_data.iloc[start_index:end_index]

            output = func(data)
            output['label'] = label
            output['chunk_counter'] = chunk['counter']
            output_data.append(output)

        return output_data

    def _update_file(self, packet: np.ndarray):
        
        for filter in self.params['filter_func'].keys():
            filtered_packet = self.params['filter_func'][filter](np.array(self.buffer)[-self.params['filter_size']:])
            chunk = self.chunk_handler[filter].process_data_packet(filtered_packet) # output dict keys: label, counter, data
            
            if not chunk['complete']:
                continue

            if self.params['save_chunks']:
                # save chunk
                pass

            for func in self.params['ml_prepare_func'].keys():
                data = self._iterate_chunk(chunk, self.params['ml_prepare_func'][func])
                data['filter'] = filter
                data['func'] = func
            
                if self.params['save_training_data']:
                    # save data
                    pass

        