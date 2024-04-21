import time
import numpy as np
import pandas as pd

from pprint import pprint
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

from eeglib.pipeline import PipelineEEG




class SessionEEG:
    
    BASE_THINKPULSE_CONFIG_GAIN_4 = [f"x{i}020010X" for i in range(1, 9)]
    BASE_THINKPULSE_CONFIG_GAIN_6 = [f"x{i}030010X" for i in range(1, 9)]
    BASE_THINKPULSE_CONFIG_GAIN_8 = [f"x{i}040010X" for i in range(1, 9)]
    # x:start_command | i: channel_index(1-8) | 0: on | 3: gain6(2 for 4, 4 for 8) | 0: adc_channel_input_source | 0: remove from BIAS | 1: SRB2 | 0: disconnect SRB1

    # config: list of channel configuration, follow https://docs.openbci.com/Cyton/CytonSDK/#channel-setting-commands
    # port: str, serial port, example: "COM3", "/dev/tty/USB0"
    def __init__(self, config=BASE_THINKPULSE_CONFIG_GAIN_6, port="COM3", simulated=False):
        self.params = BrainFlowInputParams()
        self.params.serial_port = port
        self.board = BoardShim(BoardIds.CYTON_BOARD.value if not simulated else BoardIds.SYNTHETIC_BOARD.value, self.params)
        self.config = config

    def basic_stream_start(self):
        # prepare session
        self.board.prepare_session()
        # apply channel configuration, skip if simulated
        if self.board.board_id != BoardIds.SYNTHETIC_BOARD.value:
            for conf in self.config:
                self.board.config_board(conf)
        # start stream
        self.board.start_stream()
    
    def basic_stream_stop(self):
        # finish the session
        self.board.stop_stream()
        # save data
        self.data = self.board.get_board_data()
        # release session
        self.board.release_session()

    def basic_stream(self, duration):
        self.basic_stream_start()
        time.sleep(duration)
        self.basic_stream_stop()
        
    def export_data(self, filename, write_mode="w"):
        DataFilter.write_file(self.data, filename, write_mode)
        print(f"Saved session data to {filename}\n")

    def process_pipeline(self, pipeline : PipelineEEG , timeout=0):
        # start the stream
        pipeline.prepare()
        self.basic_stream_start()
        # process the pipeline
        status = pipeline.start(self.board, timeout)
        # stop the stream
        self.basic_stream_stop()
        # return pipeline status
        return status

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

# 'Fp1','Fp2','C3','C4','P7','P8','O1','O2'