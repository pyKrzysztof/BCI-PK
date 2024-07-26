import numpy as np
import pandas as pd

from brainflow.data_filter import DataFilter, DetrendOperations, WindowOperations, FilterTypes


def filter1(data, params):
    timeseries_data = np.array(data.transpose(), order="C")

    for channel in params['channel_column_ids']:
        DataFilter.detrend(timeseries_data[channel], DetrendOperations.CONSTANT.value)
        DataFilter.perform_lowpass(timeseries_data[channel], 250, 44, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandpass(timeseries_data[channel], 250, 4.0, 45.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

    timeseries_data = timeseries_data.transpose()
    return timeseries_data

def filter2(data, params):
    timeseries_data = np.array(data.transpose(), order="C")

    for channel in params['channel_column_ids']:
        DataFilter.detrend(timeseries_data[channel], DetrendOperations.LINEAR.value)
        DataFilter.perform_lowpass(timeseries_data[channel], 250, 44, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandpass(timeseries_data[channel], 250, 4.0, 45.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

    timeseries_data = timeseries_data.transpose()
    return timeseries_data

def filter3(data, params):
    timeseries_data = np.array(data.transpose(), order="C")

    for channel in params['channel_column_ids']:
        DataFilter.detrend(timeseries_data[channel], DetrendOperations.LINEAR.value)
        DataFilter.perform_lowpass(timeseries_data[channel], 250, 44, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

    timeseries_data = timeseries_data.transpose()
    return timeseries_data

def filter4(data, params):
    timeseries_data = np.array(data.transpose(), order="C")

    for channel in params['channel_column_ids']:
        DataFilter.detrend(timeseries_data[channel], DetrendOperations.CONSTANT.value)
        DataFilter.perform_lowpass(timeseries_data[channel], 250, 44, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

    timeseries_data = timeseries_data.transpose()
    return timeseries_data

def prep1(data, params):
    temp_data = []
    # local_data = data#[[3, 4, 5, 6]]
    data_t = np.array(data.to_numpy().transpose(), order="C")

    for channel_data in data_t:
        fft_data = DataFilter.get_psd_welch(channel_data, 128, 128 // 2, 250, WindowOperations.BLACKMAN_HARRIS)
        lim = min(32, len(fft_data[0]))
        values = fft_data[0][0:lim].tolist()
        temp_data.append(values)

    return {"timeseries": data[-32:], "fftdata": pd.DataFrame(np.array(temp_data))}

def prep1_live(data, params):
    temp_data = []
    # local_data = data#[[3, 4, 5, 6]]

    data_t = np.array(data[:, params['channel_column_ids']].transpose(), order="C")
    for channel_data in data_t:
        fft_data = DataFilter.get_psd_welch(channel_data, 128, 128 // 2, 250, WindowOperations.BLACKMAN_HARRIS)
        lim = min(32, len(fft_data[0]))
        values = fft_data[0][0:lim].tolist()
        temp_data.append(values)

    return {"timeseries": data[-32:], "fftdata": pd.DataFrame(np.array(temp_data))}

def prep2(data, params):
    temp_data = []
    # local_data = data
    data_t = np.array(data[[3, 4, 5, 6]].to_numpy().transpose(), order="C")

    for channel_data in data_t:
        fft_data = DataFilter.get_psd_welch(channel_data, 128, 128 // 2, 250, WindowOperations.BLACKMAN_HARRIS)
        lim = min(32, len(fft_data[0]))
        values = fft_data[0][0:lim].tolist()
        temp_data.append(values)

    return {"timeseries": data.iloc[-32:, [3, 4, 5, 6]], "fftdata": pd.DataFrame(np.array(temp_data))}