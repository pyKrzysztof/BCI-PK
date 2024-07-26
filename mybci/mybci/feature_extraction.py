import numpy as np
import pandas as pd
import enum

from brainflow import DataFilter, DetrendOperations, FilterTypes, WaveletTypes, WaveletDenoisingTypes, ThresholdTypes, WaveletExtensionTypes, NoiseEstimationLevelTypes, WindowOperations

from mybci.data_processor import Config

class Features(enum.IntEnum):
    """Enum to store all supported extraction features. """
    TIMESERIES = 0
    FFT = 1
    BANDPOWERS = 2
    WAVELETS = 3


class Filters(enum.Enum):

    def lowpass(hf:float=45.0, order:int=2, sampling:int=250, type=FilterTypes.BUTTERWORTH_ZERO_PHASE):
        return lambda channel_data: DataFilter.perform_lowpass(channel_data, sampling, hf, order, type, 0)
    
    def bandpass(lf:float=1.0, hf:float=45.0, order:int=3, sampling:int=250, type=FilterTypes.BUTTERWORTH_ZERO_PHASE):
        return lambda channel_data: DataFilter.perform_bandpass(channel_data, sampling, lf, hf, order, type, 0)

    def bandstop(lf:float=49.5, hf:float=51.0, order:int=4, sampling:int=250, type=FilterTypes.BUTTERWORTH_ZERO_PHASE):
        return lambda channel_data: DataFilter.perform_bandstop(channel_data, sampling, lf, hf, order, type, 0)

    def wavelet_denoise(decomposition_level:int=3, threshold_type=ThresholdTypes.HARD):
        return lambda channel_data: DataFilter.perform_wavelet_denoising(channel_data, WaveletTypes.BIOR1_1, decomposition_level, WaveletDenoisingTypes.SURESHRINK, threshold_type, WaveletExtensionTypes.SYMMETRIC, NoiseEstimationLevelTypes.FIRST_LEVEL)

    def detrend(detrend_type=DetrendOperations.CONSTANT):
        return lambda channel_data: DataFilter.detrend(channel_data, detrend_type)
    
    LOWPASS = 1
    BANDPASS = 2
    BANDSTOP = 3
    WAVELET_DENOISE = 4
    DETREND = 5

filter_dict = {
    Filters.LOWPASS: Filters.lowpass,
    Filters.BANDPASS: Filters.bandpass,
    Filters.BANDSTOP: Filters.bandstop,
    Filters.WAVELET_DENOISE: Filters.wavelet_denoise,
    Filters.DETREND: Filters.detrend,
}


def get_filter_function(args):
    if not isinstance(args, list) and not isinstance(args, tuple):
        filter_func = filter_dict[args]()
    else:
        try:
            filter_func = filter_dict[args[0]](*args[1:])
        except:
            print("Wrong filter parameters / function.")
            return lambda _: _

    return filter_func


def data_filter(data: np.array, config:Config, filters:list = [], value_scale=1):
    transposed = np.array(data.transpose(), order="C")
    for channel in config.channel_column_ids:
        transposed[channel] = transposed[channel] * value_scale
        for handle in filters:
            handle(transposed[channel])

    return transposed.transpose()


def extract_features(data: pd.DataFrame, config:Config, features: list[Features], custom_features:dict = {}, filters: list = [], value_scale=1, sampling=250):
    """Extracts target features from data channels. Can also apply filters beforehand and scale the raw values. 
    Using filters here makes sense if feature extraction size and filtering size is the same."""

    transposed = np.array(data[config.channel_column_ids].to_numpy().transpose(), order="C")

    output_features = {feature.name: [] for feature in features}
    # print("TEST")
    for name in custom_features.keys():
        output_features[name] = []

    channels = range(len(transposed))
    # print(channels)
    # print(filters)

    # print(transposed.shape)
    # print(transposed.shape)
    # print(config.channel_column_ids)
    for channel in channels:
        transposed[channel] = transposed[channel] * value_scale
        for handle in filters:
            handle(transposed[channel])
            # print("Changed shit!")

        if Features.FFT in features:
            output_features[Features.FFT.name].append(np.array(DataFilter.perform_fft(transposed[channel], WindowOperations.BLACKMAN_HARRIS)))

        if Features.BANDPOWERS in features:
            psd = DataFilter.get_psd_welch(transposed[channel], 128, 64, sampling, WindowOperations.BLACKMAN_HARRIS)
            band_alpha = DataFilter.get_band_power(psd, 8.0, 12.0)
            band_mu = DataFilter.get_band_power(psd, 9.0, 13.0)
            band_beta = DataFilter.get_band_power(psd, 13.0, 30.0)
            band_low_gamma = DataFilter.get_band_power(psd, 30.0, 45.0)
            output_features[Features.BANDPOWERS.name].append(np.array([band_alpha, band_mu, band_beta, band_low_gamma]))

        if Features.WAVELETS in features:
            output_features[Features.WAVELETS.name].append(np.array(DataFilter.perform_wavelet_transform(transposed[channel], WaveletTypes.DB5, 3)[0]))

        for name, feature in custom_features:
            output_features[name].append(feature[transposed[channel]])


    # if Features.TIMESERIES in features:
    #     output_features[Features.TIMESERIES.name] = np.array(pd.DataFrame(np.array(transposed.transpose()[:, channels])))

    for name in output_features.keys():
        output_features[name] = pd.DataFrame(np.array(output_features[name]))
    # print(output_features)
    return output_features



def get_data_filter(filters:list, value_scale=1):
    return lambda data, params: data_filter(data, params, filters, value_scale)

def get_feature_extractor(features: list[Features], custom_features = {}, filters = [], value_scale = 1, sampling = 250):
    return lambda data, params: extract_features(data, params, features, custom_features, filters, value_scale, sampling)

