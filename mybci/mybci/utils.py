import time
import logging

from brainflow.data_filter import DataFilter


def calculate_time(func):

    def wrapped_func(*args, **kwargs):
        begin = time.time()
        func(*args, **kwargs)
        end = time.time()
        logging.info(f"Time in '{func.__name__}' : {(end - begin)*1000:.4f} ms")
 
    return wrapped_func


def get_fft_size_as_power_of_2(target_res, packet_size, verbose=False):
    fft_size = int(250 // target_res)
    if fft_size % packet_size != 0:
        fft_size = int(fft_size + packet_size - (fft_size % packet_size))
        # fft_size = DataFilter.get_nearest_power_of_two(fft_size)
    if verbose:
        logging.info(f"FFT resolution = {250/fft_size:.2f} Hz\nFFT data window size = {fft_size} samples")
    return fft_size