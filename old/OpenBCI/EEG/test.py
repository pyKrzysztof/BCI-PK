from lib import DataHandler, DataProcessor
from keras import models
from collections import deque
import time
import numpy as np
from brainflow import BoardShim, DataFilter, DetrendOperations, FilterTypes, WindowOperations, BrainFlowInputParams, BoardIds


sampling_rate = 250
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


def process(data, is_full):
    # print(data[0][0])
    # print(data.shape)

    # print(is_full)
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
        current_predictions = launch_prediction(prediction_packets, fft)
        # print(current_predictions)
        # print(current_predictions)
        # predictions.append(current_predictions[0][0])
        # print(sum(predictions)/len(predictions))
        predictions[0].append(current_predictions[0][0])
        predictions[1].append(current_predictions[0][1])
        print(f"{sum(predictions[0]):.2f}, {sum(predictions[1]):.2f}")
        
    # return np.array([0, 0])

    timeseries_data = timeseries_data.transpose()[-32:]
    time_delta = time.time() - time_start
    # print(time_delta)
    return timeseries_data

params = BrainFlowInputParams()
params.serial_port = "/dev/ttyUSB0"
board = BoardShim(BoardIds.CYTON_BOARD.value, params)
board.prepare_session()
board.start_stream()
# source = "data/raw_session_data/lewo_prawo_2805.csv"
source = board
processor = DataProcessor(source, 32, process, process_size=128, sep="\t", save_processed=True)
print("Starting..")
processor.process_data_sources()
board.stop_stream()
board.release_session()
print("Done.")
