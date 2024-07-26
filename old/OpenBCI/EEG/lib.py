import os
import time
import pickle
import random as rd
import pandas as pd
import numpy as np
from collections import deque

from my_utils import calculate_time
import logging

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




# Can have only one live source
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

    # @calculate_time
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
            # print(output_file)
            df.to_csv(output_file, index=False, header=False, sep=self.sep_file, float_format='%.6f')


def prepare_chunk_data(filename, output_dir, start_identifiers=[], sep='\t'):
    # Define chunk end identifiers
    end_identifiers = [-x for x in start_identifiers]

    # Read the CSV file
    df = pd.read_csv(filename, sep=sep)

    # Variables to keep track of the current chunk and counters for each identifier
    current_chunk = []
    counters = {start: 0 for start in start_identifiers}

    # Iterate through the DataFrame
    for index, row in df.iterrows():
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
                identifier = start_identifiers.index(start_identifier) + 1
                chunk_df = pd.DataFrame(current_chunk)
                filename_prefix = f"{int(start_identifier)}_{counters[start_identifier]}"
                chunk_df.to_csv(os.path.join(output_dir, f'{filename_prefix}.csv'), header=False, index=False, sep=sep)
                print(f"{filename_prefix} processed.")
                counters[start_identifier] += 1
                current_chunk = []
        elif current_chunk:
            # If in a chunk, add the row to the current chunk
            current_chunk.append(row)


def load_data_set(name, split=0.2):
    train_data = {}
    test_data = {}
    path = os.path.join("data/", "training/", name)
    for file in os.listdir(path):
        if not file.endswith(".pickle"):
            continue
    
        marker = file.split('.')[0]
        with open(os.path.join(path, file), 'rb') as f:
            temp = pickle.load(f)
            rd.shuffle(temp)
            split_idx = int(len(temp)*split)
            print(split_idx)
            train_data[marker] = temp[:-split_idx]
            test_data[marker] = temp[-split_idx:]

    return train_data, test_data















if __name__ == "__main__":

    from brainflow import DataFilter, DetrendOperations, FilterTypes, WindowOperations
    sampling_rate = 250
    from keras import models
    model : models.Model = models.load_model("models/32_128_model_2.keras")

    predictions = [deque(maxlen=8), deque(maxlen=8)]
    # predictions = deque(maxlen=8)

    def launch_prediction(timeseries, fft):
        # print(timeseries.shape, fft.shape)
        assert timeseries.shape == (32, 8)
        assert fft.shape == (8, 32)
        reshaped_timeseries = timeseries.reshape(1, 32, 8, 1)
        reshaped_fft = fft.reshape(1, 8, 32, 1)
        return model.predict([reshaped_fft, reshaped_timeseries])


    def process(data, is_full=True):
        # timeseries_data = np.array(data[-32:, :].transpose(), order="C")
        timeseries_data = np.array(data.transpose(), order="C")
        time_start = time.time()
        for channel in [1, 2, 3, 4, 5, 6, 7, 8]:
            DataFilter.detrend(timeseries_data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(timeseries_data[channel], sampling_rate, 4.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(timeseries_data[channel], sampling_rate, 48.0, 52.0, 2,
                                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(timeseries_data[channel], sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

        prediction_packets = timeseries_data.transpose()[-32: , 1:9]

        # true when data of length process_size was passed
        if is_full:
            # calculate fft data
            data = []
            X = timeseries_data.shape[1]
            if X != 128:
                return np.array([0, 0])
            for channel in range(1, 9):
                fft_data = DataFilter.get_psd_welch(timeseries_data[channel], X, X // 2, sampling_rate, WindowOperations.BLACKMAN_HARRIS)
                lim = min(32, len(fft_data[0]))
                values = fft_data[0][0:lim].tolist()
                # data[0] = fft_data[1][0:lim].tolist() # xvalues
                data.append(values)

            fft = np.array(data)
            # current_predictions = launch_prediction(prediction_packets, fft)
            # print(current_predictions)
            # predictions.append(current_predictions[0][0])
            # print(sum(predictions)/len(predictions))
            # predictions[0].append(current_predictions[0][0])
            # predictions[1].append(current_predictions[0][1])
            # print(sum(predictions[0]), sum(predictions[1]))
            
        # return np.array([0, 0])

        timeseries_data = timeseries_data.transpose()[-32:]
        time_delta = time.time() - time_start
        # print(time_delta)
        return timeseries_data

    file_path = "data/raw_session_data/1.csv"
    data_sources = [file_path, ]
    processor = DataProcessor(data_sources, 32, process, process_size=128, sep="\t", save_processed=True)
    print("Starting..")
    processor.process_data_sources()
    print("Done.")

