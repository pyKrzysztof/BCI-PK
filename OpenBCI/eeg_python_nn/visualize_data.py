import logging
import argparse
import numpy as np
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations

from data_visualization import AppAnalyzer, AppBase, AppPresenter
from my_utils import calculate_time, get_fft_size


# raw data
def process_1(data, channel, sampling_rate):
    return data

# detrend data
def process_2(data, channel, sampling_rate):
    DataFilter.detrend(data[channel], DetrendOperations.LINEAR.value)
    return data

# detrend + bandstops at e.freq
def process_3(data, channel, sampling_rate):
    DataFilter.detrend(data[channel], DetrendOperations.LINEAR.value)

    DataFilter.perform_bandstop(data[channel], sampling_rate, 48.0, 52.0, 2,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
    DataFilter.perform_bandstop(data[channel], sampling_rate, 58.0, 62.0, 2,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
    return data

# detrend + bandpass + bandstops
def process_4(data, channel, sampling_rate):
    DataFilter.detrend(data[channel], DetrendOperations.LINEAR.value)
    DataFilter.perform_bandpass(data[channel], sampling_rate, 4.0, 45.0, 2,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
    DataFilter.perform_bandstop(data[channel], sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
    DataFilter.perform_bandstop(data[channel], sampling_rate, 58.0, 62.0, 2,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
    return data


processes = [process_1, process_2, process_3, process_4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", required=False, help="Device port.")
    parser.add_argument("--from-file", type=str, default="", required=False, help="Runs .csv data playback instead of a live board connection.")
    parser.add_argument("--app-id", type=int, default=0, required=False, help="Choose application to use, 0 for default visualization app on all channels, 1 for detailed comparison of data between processes (single channel)")
    parser.add_argument("-simulated", action="store_true", required=False, help="If --from-file source was NOT specified this will simulate a live board data.")
    parser.add_argument("--packet-size", type=int, default=64, required=False, help="Amount of data packets to be analyzed at a time by the update() method and processes.")
    parser.add_argument("--update-ms", type=int, default=100, required=False, help="Interval of running the internal update() method.")
    parser.add_argument("--display-time", type=int, default=100, required=False, help="Time window of the recent data to be displayed, independantly of fft calculation.")
    parser.add_argument("-debug-time", action="store_true", required=False)
    parser.add_argument("--window", type=int, default=0, required=False, help="Window method for FFT calculation, 0 = BLACKMAN_HARRIS, 1 = HAMMING")
    parser.add_argument("--fft-res", type=float, default=2.0, required=False, help="FFT resolution [Hz]")
    args = parser.parse_args()
    
    fft_size = get_fft_size(args.fft_res, args.packet_size, verbose=True)

    kwargs = {"board_shim": None, "from_file": args.from_file, "points_to_analyze": args.packet_size, 
              "update_speed_ms": args.update_ms, "display_s": args.display_time, "fft_size": fft_size,
              "processes": processes}

    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)
    
    params = BrainFlowInputParams()
    params.serial_port = args.port

    app_handle = AppPresenter if args.app_id == 1 else AppAnalyzer

    if args.window == 1:
        AppBase.window_method = WindowOperations.HAMMING
    else: 
        AppBase.window_method = WindowOperations.BLACKMAN_HARRIS

    if args.debug_time:
        app_handle.update_file = calculate_time(app_handle.update_file)
        app_handle.update_live = calculate_time(app_handle.update_live)

    # run from file
    if args.from_file != "":
        return app_handle(**kwargs)
    
    # run from live data
    if args.simulated:
        board_id = BoardIds.SYNTHETIC_BOARD
    else:
        board_id = BoardIds.CYTON_BOARD

    board = BoardShim(board_id=board_id, input_params=params)
    board.prepare_session()
    board.start_stream(60*250) # ringbuffer for 60 seconds of data.
    kwargs["board_shim"] = board
    app = app_handle(**kwargs)

    try:
        if board.is_prepared():
            board.release_session()
        logging.info("Released session.")
    except Exception as e:
        logging.warning("Exception", exc_info=True)


if __name__ == "__main__":
    main()
