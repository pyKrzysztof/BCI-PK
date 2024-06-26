import mybci

import numpy as np
import pandas as pd

from brainflow.data_filter import DataFilter, DetrendOperations, WindowOperations, FilterTypes

def filter_func_1(data, params):
    timeseries_data = np.array(data[-32:].transpose(), order="C")
    
    for channel in params['channel_column_ids']:
        DataFilter.detrend(timeseries_data[channel], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(timeseries_data[channel], 250, 4.0, 45.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

    timeseries_data = timeseries_data.transpose()#[-32:]
    return timeseries_data

def ml_prep(data, params):
    temp_data = []
    # print(data.columns)
    local_data = data[[3, 4, 5, 6]]
    data_t = np.array(local_data.to_numpy().transpose(), order="C")
    # print(data_t.shape)
    for channel_data in data_t:
        fft_data = DataFilter.get_psd_welch(channel_data, 128, 128 // 2, 250, WindowOperations.BLACKMAN_HARRIS)
        lim = min(32, len(fft_data[0]))
        values = fft_data[0][0:lim].tolist()
        temp_data.append(values)

    fft = pd.DataFrame(np.array(temp_data))
    return {"timeseries": local_data[-32:], "fftdata": fft}

config = mybci.get_base_config()

config['name'] = "LRClassification2"
config['session_file'] = []
config['session_file'].extend([f'data/session/0406_{i}.csv' for i in [1, 2]])
config['session_file'].extend([f'data/session/1806_{i}.csv' for i in range(1, 6)])
config['session_file'].extend([f'data/session/2006_{i}.csv' for i in range(1, 6)])
config['action_markers'] = [1, 2]
config['filter_func'] = {'F1': filter_func_1} #, 'F4': filter_func_4}
config['ml_prepare_func'] = {'ML1': ml_prep}
config['save_training_dataset'] = True
config['keep_seperate_training_data'] = False
config['output_path_training_dataset'] = "data/datasets/"

processor = mybci.DataProcessor(config)
processor.process()
print("Complete.")
