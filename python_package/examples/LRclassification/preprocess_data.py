import mybci

import numpy as np
import pandas as pd

from brainflow.data_filter import DataFilter, DetrendOperations, WindowOperations, FilterTypes



def filter_func_1(data, params):
    timeseries_data = np.array(data.transpose(), order="C")
    
    for channel in params['channel_column_ids']:
        DataFilter.detrend(timeseries_data[channel], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(timeseries_data[channel], 250, 4.0, 45.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

    timeseries_data = timeseries_data.transpose()[-32:]
    return timeseries_data

def filter_func_2(data, params):
    timeseries_data = np.array(data.transpose(), order="C")
    
    for channel in params['channel_column_ids']:
        DataFilter.detrend(timeseries_data[channel], DetrendOperations.LINEAR.value)
        DataFilter.perform_bandpass(timeseries_data[channel], 250, 4.0, 45.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

    timeseries_data = timeseries_data.transpose()[-32:]
    return timeseries_data

def filter_func_3(data, params):
    timeseries_data = np.array(data.transpose(), order="C")
    
    for channel in params['channel_column_ids']:
        DataFilter.detrend(timeseries_data[channel], DetrendOperations.CONSTANT.value)
        DataFilter.perform_lowpass(timeseries_data[channel], 250, 40, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandpass(timeseries_data[channel], 250, 4.0, 40.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

    timeseries_data = timeseries_data.transpose()[-32:]
    return timeseries_data

def filter_func_4(data, params):
    timeseries_data = np.array(data.transpose(), order="C")
    
    for channel in params['channel_column_ids']:
        DataFilter.detrend(timeseries_data[channel], DetrendOperations.CONSTANT.value)
        DataFilter.perform_lowpass(timeseries_data[channel], 250, 40, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

    timeseries_data = timeseries_data.transpose()[-32:]
    return timeseries_data

def ml_prep(data, params):
    temp_data = []
    data_t = np.array(data.to_numpy().transpose(), order="C")
    for channel_data in data_t:
        fft_data = DataFilter.get_psd_welch(channel_data, 128, 128 // 2, 250, WindowOperations.BLACKMAN_HARRIS)
        lim = min(32, len(fft_data[0]))
        values = fft_data[0][0:lim].tolist()
        temp_data.append(values)

    fft = pd.DataFrame(np.array(temp_data))
    return {"timeseries": data[-32:], "fftdata": fft}

def prediction_function_1(data):
    return 0

config = mybci.get_base_config()

config['name'] = "LRClassification"
config['session_file'] = ['data/session/0406_1.csv', 'data/session/0406_2.csv', 'data/session/2805.csv']
config['action_markers'] = [1, 2]
config['filter_func'] = {'F1': filter_func_1, 'F2': filter_func_2, 'F3': filter_func_3, 'F4': filter_func_4}
config['ml_prepare_func'] = {'ML1': ml_prep}
config['save_training_dataset'] = True
config['keep_seperate_training_data'] = False
# config['use_board_device'] = True
# config['prediction_functions'] = {"pf1": prediction_function_1}
# processor.update()


processor = mybci.DataProcessor(config)
processor.process()

print("Complete.")
