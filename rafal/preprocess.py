import numpy as np
import pandas as pd

from brainflow.data_filter import DataFilter, DetrendOperations, WindowOperations, FilterTypes, WaveletTypes, ThresholdTypes, WaveletDenoisingTypes, WaveletExtensionTypes, NoiseEstimationLevelTypes


def filter1(data, params):
    timeseries_data = np.array(data.transpose(), order="C")

    for channel in params['channel_column_ids']:
        timeseries_data[channel] = timeseries_data[channel]*1e-6
        # DataFilter.detrend(timeseries_data[channel], DetrendOperations.CONSTANT.value) # dont do this??
        # DataFilter.perform_lowpass(timeseries_data[channel], 250, 44, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0) # dont do this 
        DataFilter.perform_bandpass(timeseries_data[channel], 250, 1.0, 45.0, 3, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(timeseries_data[channel], 250, 49.5, 51, 4, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

        DataFilter.perform_wavelet_denoising(timeseries_data[channel], WaveletTypes.BIOR1_1, 3,
                                            WaveletDenoisingTypes.SURESHRINK, ThresholdTypes.HARD,
                                            WaveletExtensionTypes.SYMMETRIC, NoiseEstimationLevelTypes.FIRST_LEVEL)

    timeseries_data = timeseries_data.transpose()
    return timeseries_data


def prep1(data, params):
    temp_data = []
    # local_data = data#[[3, 4, 5, 6]]
    data_t = np.array(data.to_numpy().transpose(), order="C")
    beta_bandpowers = []
    for channel_data in data_t:
        psd = DataFilter.get_psd_welch(channel_data, 128, 128 // 2, 250, WindowOperations.BLACKMAN_HARRIS)
        lim = min(32, len(psd[0]))
        values = psd[0][0:lim].tolist()
        temp_data.append(values)

        band_power_smr = DataFilter.get_band_power(psd, 13.0, 15.0)
        band_power_beta = DataFilter.get_band_power(psd, 13.0, 30.0)
        beta_bandpowers.append(band_power_beta, )

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
