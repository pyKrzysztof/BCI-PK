"""
Need to setup serial port binding from Windows to WSL2 if running the code in WSL and not native Linux.

On Windows, as an administrator download (usbipd) and run:
usbipd list (find busid for the openbci dongle)
usbipd bind --busid <busid>
    Note that as long as the USB device is attached to WSL, it cannot be used by Windows.
    Ensure that a WSL command prompt is open in order to keep the WSL 2 lightweight VM active.
usbipd attach --wsl --busid <busid>

To detach: (or just disconnect the device)
usbipd detach --busid <busid>

On WSL2:
lsusb
"""

"""
> BoardShim.get_board_descr(BOARD)
{'accel_channels': [9, 10, 11],
 'analog_channels': [19, 20, 21],
 'ecg_channels': [1, 2, 3, 4, 5, 6, 7, 8],
 'eeg_channels': [1, 2, 3, 4, 5, 6, 7, 8],
 'eeg_names': 'Fp1,Fp2,C3,C4,P7,P8,O1,O2',
 'emg_channels': [1, 2, 3, 4, 5, 6, 7, 8],
 'eog_channels': [1, 2, 3, 4, 5, 6, 7, 8],
 'marker_channel': 23,
 'name': 'Cyton',
 'num_rows': 24,
 'other_channels': [12, 13, 14, 15, 16, 17, 18],
 'package_num_channel': 0,
 'sampling_rate': 250,
 'timestamp_channel': 22}

Channel names list: 'Fp1','Fp2','C3','C4','P7','P8','O1','O2'
"""
from pprint import pprint
import datetime
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter

# simulation flag
SIMULATED = True 

# 8 electrodes, Cyton board, serial connection through openbci dongle
ELECTRODES = range(1, 9)
BOARD = BoardIds.CYTON_BOARD
PORT = "/dev/ttyS3"

# base configurations for thinkpulse active electrodes
BASE_THINKPULSE_CONFIG_GAIN_1 = [f"x{i}000010X" for i in ELECTRODES]
BASE_THINKPULSE_CONFIG_GAIN_2 = [f"x{i}010010X" for i in ELECTRODES]
BASE_THINKPULSE_CONFIG_GAIN_4 = [f"x{i}020010X" for i in ELECTRODES]
BASE_THINKPULSE_CONFIG_GAIN_6 = [f"x{i}030010X" for i in ELECTRODES]
BASE_THINKPULSE_CONFIG_GAIN_8 = [f"x{i}040010X" for i in ELECTRODES]

params = BrainFlowInputParams()
params.serial_port = PORT
board = BoardShim(BOARD.value if not SIMULATED else BoardIds.SYNTHETIC_BOARD.value, params)

package_num_channel = BoardShim.get_package_num_channel(board.board_id)
eeg_channels = BoardShim.get_eeg_channels(board.board_id)
marker_channel = BoardShim.get_marker_channel(board.board_id)

board.prepare_session()

config = "".join(BASE_THINKPULSE_CONFIG_GAIN_8)
board.config_board(config)
current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H%M%S")
board.add_streamer(f"file://{current_time}_default.csv:w", preset=BrainFlowPresets.DEFAULT_PRESET)
board.add_streamer(f"file://{current_time}_aux.csv:w", preset=BrainFlowPresets.AUXILIARY_PRESET)
board.start_stream()

start_time = datetime.datetime.now()
endTime = start_time + datetime.timedelta(seconds=5)
markerTime = start_time + datetime.timedelta(milliseconds=500)
markerEndTime = markerTime + datetime.timedelta(milliseconds=500)
previous_buffer_count = 0

while True:
    if datetime.datetime.now() >= endTime:
        break
    
    # get buffer count
    buffer_count = board.get_board_data_count()
    
    # skip if no new packets in the buffer
    if previous_buffer_count == buffer_count:
        continue
    
    # get last 50 packets (those are not removed from the buffer, just copied) just because I can
    # I might be able to pass this data and make live predictions with a trained model,
    # not removing previous packets might be adventagous, 
    # it would allow to make more predictions with less delay and with possible corrections.
    packets = board.get_current_board_data(50)
    
    # insert markers through a specific timeframe
    now = datetime.datetime.now()
    if markerTime <= now <= markerEndTime:
            board.insert_marker(buffer_count)

    # update the previous_buffer_count
    previous_buffer_count = buffer_count

board.stop_stream()

full_data = board.get_board_data()

board.release_session()
