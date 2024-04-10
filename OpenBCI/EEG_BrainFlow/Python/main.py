import argparse
import time
import numpy as np
import pandas as pd

from pprint import pprint
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter


# temp
def in_session(duration):
    time.sleep(duration)


# x:start_command | i: channel_index(1-8) | 0: on | 3: gain6(2 for 4, 4 for 8) | 0: adc_channel_input_source | 0: remove from BIAS | 1: SRB2 | 0: disconnect SRB1
BASE_THINKPULSE_CONFIG_GAIN_4 = [f"x{i}020010X" for i in range(1, 9)]
BASE_THINKPULSE_CONFIG_GAIN_6 = [f"x{i}030010X" for i in range(1, 9)]
BASE_THINKPULSE_CONFIG_GAIN_8 = [f"x{i}040010X" for i in range(1, 9)]

class SessionEEG:
    
    # filename: str, example: "file.csv"
    # training_pipeline: list, example: [(func1, time1), (func2, time2), ...] #TODO
    # config: list of channel configuration, follow https://docs.openbci.com/Cyton/CytonSDK/#channel-setting-commands
    # write_mode: str, can be "append" for extended sessions or "write" for single session files.
    # port: str, serial port, example: "COM3", "/dev/tty/USB0"
    # duration: num, session duration in seconds
    def __init__(self, filename, training_pipeline, config=BASE_THINKPULSE_CONFIG_GAIN_6, write_mode="append", port="COM3", duration=10):
        self.params = BrainFlowInputParams()
        self.params.serial_port = port
        self.filename = filename
        self.duration = duration
        self.board = BoardShim(BoardIds.CYTON_BOARD.value, self.params)
        self.config = config
        self.pipeline = training_pipeline
        self.write_mode = "a" if write_mode == "append" else "w"
        
        
    def start_session(self):
        # prepare session
        self.board.prepare_session()
        # apply channel configuration
        for conf in self.config:
            self.board.config_board(conf)
        # start stream
        self.board.start_stream()
        
        # timing the session, process the training pipeline and insert markers
        in_session(self.duration)
        
        # finish the session
        self.board.stop_stream()
        
        # save data
        self.data = self.board.get_board_data()
        DataFilter.write_file(self.data, self.filename, self.write_mode)
        
        # release session
        self.board.release_session()
    
    def process_data(self):
        self.data_df = pd.DataFrame(np.transpose(self.data))
        print(self.data_df)


def restore_data(filename):
    restored_data = DataFilter.read_file(filename)
    return pd.DataFrame(np.transpose(restored_data))


def board_info():
    board_id = BoardIds.CYTON_BOARD.value
    pprint(BoardShim.get_board_descr(board_id))

#  'eeg_channels': [1, 2, 3, 4, 5, 6, 7, 8],
#  'eeg_names': 'Fp1,Fp2,C3,C4,P7,P8,O1,O2',
#  'emg_channels': [1, 2, 3, 4, 5, 6, 7, 8],
#  'eog_channels': [1, 2, 3, 4, 5, 6, 7, 8],
#  'marker_channel': 23,
#  'name': 'Cyton',
#  'num_rows': 24,
#  'other_channels': [12, 13, 14, 15, 16, 17, 18],
#  'package_num_channel': 0,
#  'sampling_rate': 250,
#  'timestamp_channel': 22}