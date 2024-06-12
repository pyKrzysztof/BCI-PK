import time

import numpy as np
import pandas as pd

from collections import deque



class DataHandler:
    def __init__(self, data_source, packet_size, is_live=False, sep=','):
        self.data_source = data_source
        self.is_live = is_live
        self.packet_size = packet_size
        self.sep=sep

        if self.is_live:
            from brainflow.board_shim import BoardShim
            if not isinstance(self.data_source, BoardShim):
                raise ValueError("For live mode, data_source must be an instance of BoardShim")
            self.board = self.data_source
            self.get_data_packet = self._get_live_data
        else:
            if not isinstance(self.data_source, str):
                raise ValueError("For non-live mode, data_source must be a file path")
            self.filepath = self.data_source
            self.data_iter = self._file_data_generator()
            self.get_data_packet = self._get_file_data

    def _file_data_generator(self):
        for chunk in pd.read_csv(self.filepath, chunksize=self.packet_size, sep=self.sep):
            yield chunk.values

    def _get_live_data(self):
        if self.board.get_board_data_count() >= self.packet_size:
            data = self.board.get_board_data(self.packet_size).transpose()
            # print(data.shape, data[0][0])
            return data
        else:
            return np.array([]) # Not enough data available

    def _get_file_data(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            return np.array([])  # No more data in the file




class DataProcessor:
    def __init__(self, data_source, packet_size, process_function, process_size, output_file=None, timeout=10, sep=',', save_processed=True):
        self.data_source = data_source
        self.packet_size = packet_size
        self.process_function = process_function
        self.process_size = process_size
        self.timeout = timeout
        self.packet_buffer = deque(maxlen=self.process_size // self.packet_size)
        self.sep_file = sep
        self.save_processed = save_processed
        self.output_file = output_file

    def process_data_sources(self):
        if isinstance(self.data_source, str):
            self._process_file_source()
        else:
            self._process_live_source()

    def _process_file_source(self):
        handler = DataHandler(self.data_source, self.packet_size, False, sep=self.sep_file)
        processed_data = []
        while True:
            data_packet = handler.get_data_packet()
            if data_packet.size == 0:
                break
            self.packet_buffer.append(data_packet)
            combined_packets = np.vstack(self.packet_buffer)
            processed_packet = self.process_function(combined_packets, len(self.packet_buffer) * self.packet_size >= self.process_size)
            processed_data.append(processed_packet)
        print("Saving..")
        self._save_processed_data(handler, processed_data)

    def _process_live_source(self):
        handler = DataHandler(self.data_source, self.packet_size, True)
        processed_data = []
        start_time = time.time()
        while True:
            data_packet = handler.get_data_packet()
            current_time = time.time()
            if data_packet.size == self.packet_size:
                self.packet_buffer.append(data_packet)
                start_time = current_time
                combined_packets = np.vstack(self.packet_buffer)
                processed_packet = self.process_function(combined_packets, len(self.packet_buffer) * self.packet_size >= self.process_size)
                if self.save_processed:
                    processed_data.append(processed_packet)
            elif current_time - start_time > self.timeout:
                break  # Timeout
            else:
                pass
        self._save_processed_data(handler, processed_data)

    def _save_processed_data(self, handler, processed_data):
        if handler.is_live:
            output_file = self.output_file
        else:
            if self.output_file == None:
                output_file = handler.filepath.replace('.csv', '_processed.csv')
            else:
                output_file = self.output_file
        if processed_data:
            processed_data_array = np.vstack(processed_data[:-1])
            df = pd.DataFrame(processed_data_array)
            df.to_csv(output_file, index=False, header=False, sep=self.sep_file, float_format='%.6f')
