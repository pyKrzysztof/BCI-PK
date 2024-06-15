import mybci

import numpy as np
import pandas as pd

from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations



def filter_func_1(data: np.ndarray[any, np.dtype], params: dict[str, any]) -> np.ndarray[any, np.dtype]:
    # data is of size "process_size"
    timeseries_data = np.array(data.transpose(), order="C")
    
    for channel in range(data.shape[1]):
        DataFilter.detrend(timeseries_data[channel], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(timeseries_data[channel], 250, 4.0, 45.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 48.0, 52.0, 2,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 58.0, 62.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

    timeseries_data = timeseries_data.transpose()[-32:]
    return timeseries_data


def ml_prepare_func_1(data: pd.DataFrame, params: dict[str, any]) -> dict[str, pd.DataFrame]:
    # data is of size "ml_prepare_size"
    output_data = {"timeseries": None, "fftdata": None}
    output_data["timeseries"] = data[-32:]

    # fft data
    data = [[], ]
    for channel_data in data:
        fft_data = DataFilter.get_psd_welch(channel_data, 128, 128 // 2, 250, WindowOperations.BLACKMAN_HARRIS)
        lim = min(32, len(fft_data[0]))
        values = fft_data[0][0:lim].tolist()
        data[0] = fft_data[1][0:lim].tolist()
        data.append(values)

    fft = pd.DataFrame(np.array(data))
    output_data["fftdata"] = fft

    # returns training data for the packet.
    return output_data


config = {
    # live board data config - not implemented yet.
    "board_device": "/dev/ttyUSB0", # board connection physical port.
    "use_board_device": False, # whether to use the board for processing (toggles live data processing / file data processing).
    "save_raw_session": True, # when in live data processing, toggles raw session data file saving.
    "live_pipeline": None, # when in live data processing mode, specifies (optionally) a pipeline for training, blocks live prediction.
    "prediction_functions": [], # functions that take current training packet data (output of ml_prepare_func)
    
    # File playback config
    "session_file": "data/session/my_session_data.csv",
    
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
    # "ml_prepare_func": {"data_1": ml_prepare_func_1}, # a dictionary of functions that have to return a dictionary of ""file type": data" key-value pairs for passed data, the output data will be saved and passed as ML data.
    "ml_prepare_func": {"model2set": ml_prepare_func_1},
    "ml_prepare_extra_columns": [],
    # example usage would be generating fftdata from timeseries data and saving both data to be used for training.
    # multiple functions can be passed for creating many datasets with different data.

    # custom data processing config - only makes sense for pipelined / labeled data.
    "chunk_func_pass_extra_columns": [], # extra columns to pass to the chunk processing function, eg. accel channels.
    "chunk_func": None, # a function that gets passed COMPLETE single chunk data (from live pipeline processing or offline processing) to be used when writing custom behaviour. 

    # filesystem config
    "output_path_chunks": "data/chunks/my_session/",
    "output_path_training_data": "data/training/my_session/",
    "output_path_training_dataset": "data/datasets/my_session/",
    "save_chunks": True, # saves chunks to a folder, could be useful when making many changes to processing functions on huge amount of data.
    "keep_seperate_training_data": False, # if True, it will not remove the intermediate files for training dataset making.
    "save_training_dataset": True, # it"s possible to disable it but why would you use this processor then.
    "sep": "\t", # value separator in data files.
}


processor = mybci.NewDataProcessor(config)

while processor.update():
    pass

print("Complete.")
