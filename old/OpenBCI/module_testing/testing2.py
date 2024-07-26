import mybci

import numpy as np
import pandas as pd

from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations



def filter_func_1(data: np.ndarray[any, np.dtype], params: dict[str, any]) -> np.ndarray[any, np.dtype]:
    # data is of size "process_size"
    timeseries_data = np.array(data.transpose(), order="C")
    
    for channel in params['channel_column_ids']:
        DataFilter.detrend(timeseries_data[channel], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(timeseries_data[channel], 250, 4.0, 45.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

    timeseries_data = timeseries_data.transpose()[-32:]
    return timeseries_data


def ml_prepare_func_1(data: pd.DataFrame, params: dict[str, any]) -> dict[str, pd.DataFrame]:
    # data is of size "ml_prepare_size"

    temp_data = []
    data_t = np.array(data.to_numpy().transpose(), order="C")
    for channel_data in data_t:
        fft_data = DataFilter.get_psd_welch(channel_data, 128, 128 // 2, 250, WindowOperations.BLACKMAN_HARRIS)
        lim = min(32, len(fft_data[0]))
        values = fft_data[0][0:lim].tolist()
        temp_data.append(values)

    fft = pd.DataFrame(np.array(temp_data))

    return {"timeseries": data[-32:], "fftdata": fft}


config = {
    # live board data config - not implemented yet.
    "board_device": "/dev/ttyUSB0", # board connection physical port.
    "use_board_device": False, # whether to use the board for processing (toggles live data processing / file data processing).
    "save_raw_session": True, # when in live data processing, toggles raw session data file saving.
    "live_pipeline": None, # when in live data processing mode, specifies (optionally) a pipeline for training, blocks live prediction.
    "prediction_functions": [], # functions that take current training packet data (output of ml_prepare_func)
    
    # File playback config
    "session_file": "",
    "name": "",
    # common data processing config
    "action_markers": [1, 2], # markers of the data to be processed.
    "buffer_size": 128, # data buffer size, for default behaviour should be the biggest of the three (packet_size, filter_size, ml_prepare_size)
    "channel_column_ids": [1, 2, 3, 4, 5, 6, 7, 8], # dataframe / array columns to be passed to the filtering function.
    "packet_size": 32, # how many data samples are to be received at a time.
    "filter_size": 128, # amount of packets to be passed to the filter function.
    "filter_func": {"filtered1": filter_func_1}, # can return any amount of data rows but only last "packet_size" entries will be saved. Multiple filtering functions can be passed to generate more data.
    "filter_func_extra_columns": [], # extra columns to pass to the filtering function, eg. accel channels.
    "ml_prepare_size": 128, # amount of samples to be sent to the ml_prepare_func
    "ml_prepare_chunk_start_offset": 250, # amount of samples to pass after chunk starts before starting the "ml_prepare_func" (eg. if the chunk starts at packet no. 0, the first call to ml_prepare_func will be with packets no. 0 + "ml_prepare_chunk_start_offset" - "ml_prepare_size". Main use is consideration for human reaction time in training.
    "ml_prepare_func": {"mlprep1": ml_prepare_func_1}, # a dictionary of functions that have to return a dictionary of ""file type": data" key-value pairs for passed data, the output data will be saved and passed as ML data.
    "ml_prepare_extra_columns": [],
    # example usage would be generating fftdata from timeseries data and saving both data to be used for training.
    # multiple functions can be passed for creating many datasets with different data.

    # custom data processing config - only makes sense for pipelined / labeled data.
    "chunk_func_pass_extra_columns": [], # extra columns to pass to the chunk processing function, eg. accel channels.
    "chunk_func": None, # a function that gets passed COMPLETE single chunk data (from live pipeline processing or offline processing) to be used when writing custom behaviour. 

    # filesystem config
    "output_path_chunks": "data/chunks/",
    "output_path_training_data": "data/training/",
    "output_path_training_dataset": "data/datasets/",
    "save_chunks": False, # saves chunks to a folder, could be useful when making many changes to processing functions on huge amount of data.
    "keep_seperate_training_data": False, # if True, it will not remove the intermediate files for training dataset making.
    "save_training_dataset": True, # it"s possible to disable it but why would you use this processor then.
    "sep": "\t", # value separator in data files.
}


config['session_file'] = "data/session/0406_1.csv"
config['name'] = "0406_1"
processor = mybci.NewDataProcessor(config)
while processor.update():
    pass
print("Done 1/3")

config['session_file'] = "data/session/0406_2.csv"
config['name'] = "0406_2"
processor = mybci.NewDataProcessor(config)
while processor.update():
    pass
print("Done 2/3")

config['session_file'] = "data/session/2805.csv"
config['name'] = "2805"
processor = mybci.NewDataProcessor(config)
while processor.update():
    pass
print("Done 3/3")

input_file = "data/datasets/0406_1_filtered1_model2set.pickle"
train_data_1, test_data_1 = mybci.load_and_split_data(input_file, 0.2, load_all=False)
input_file = "data/datasets/0406_2_filtered1_model2set.pickle"
train_data_2, test_data_2 = mybci.load_and_split_data(input_file, 0.2, load_all=False)
input_file = "data/datasets/2805_filtered1_model2set.pickle"
train_data_3, test_data_3 = mybci.load_and_split_data(input_file, 0.2, load_all=False)

from keras import models
import random as rd

model : models.Model = models.load_model("models/base/32_128_model_2.keras")

# training
train_data = []
train_data.extend(train_data_1)
train_data.extend(train_data_2)
train_data.extend(train_data_3)
rd.shuffle(train_data)
model : models.Model = models.load_model("models/base/32_128_model_2.keras")
X1 = np.array([data['fftdata'] for data in train_data])
X2 = np.array([data['timeseries'] for data in train_data])
Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in train_data])
model.fit([X1, X2], Y, batch_size=32, epochs=50, validation_split=0.2)

# testing
def test_count(data, idx_zero, idx_non_zero):
    test = [0, 0]
    for a, b in data:
        test[0] = test[0] + a
        test[1] = test[1] + b
    assert test[idx_zero] == 0
    assert test[idx_non_zero] != 0

X1 = np.array([data['fftdata'] for data in test_data_1[1]])
X2 = np.array([data['timeseries'] for data in test_data_1[1]])
Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data_1[1]])
test_count(Y, 1, 0)
model.evaluate([X1, X2], Y)

X1 = np.array([data['fftdata'] for data in test_data_1[2]])
X2 = np.array([data['timeseries'] for data in test_data_1[2]])
Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data_1[2]])
test_count(Y, 0, 1)
model.evaluate([X1, X2], Y)

X1 = np.array([data['fftdata'] for data in test_data_2[1]])
X2 = np.array([data['timeseries'] for data in test_data_2[1]])
Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data_2[1]])
test_count(Y, 1, 0)
model.evaluate([X1, X2], Y)

X1 = np.array([data['fftdata'] for data in test_data_2[2]])
X2 = np.array([data['timeseries'] for data in test_data_2[2]])
Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data_2[2]])
test_count(Y, 0, 1)
model.evaluate([X1, X2], Y)

X1 = np.array([data['fftdata'] for data in test_data_3[1]])
X2 = np.array([data['timeseries'] for data in test_data_3[1]])
Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data_3[1]])
test_count(Y, 1, 0)
model.evaluate([X1, X2], Y)

X1 = np.array([data['fftdata'] for data in test_data_3[2]])
X2 = np.array([data['timeseries'] for data in test_data_3[2]])
Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data_3[2]])
test_count(Y, 0, 1)
model.evaluate([X1, X2], Y)
