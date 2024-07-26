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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
timestamp_channel = BoardShim.get_timestamp_channel(board.board_id)

MAX_PLOT_SIZE = 500
xs = [i for i in range (MAX_PLOT_SIZE)]
refresh_rate = 1/BoardShim.get_sampling_rate(board.board_id)

import time
sinceEpochStart = 0

fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(4, 8), constrained_layout=True)
fig.canvas.draw()
plt.pause(1)

def update_plot(y):
    relative_times = [t[-1] - sinceEpochStart for t in y]
    for i, ax in enumerate(axes):
        ax.clear()
        ax.plot(relative_times, [channel[i] for channel in y])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    fig.canvas.draw()
    plt.pause(refresh_rate)


board.prepare_session()

config = "".join(BASE_THINKPULSE_CONFIG_GAIN_8)
board.config_board(config)
current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H%M%S")
# board.add_streamer(f"file://{current_time}_default.csv:w", preset=BrainFlowPresets.DEFAULT_PRESET)
# board.add_streamer(f"file://{current_time}_aux.csv:w", preset=BrainFlowPresets.AUXILIARY_PRESET)

time.sleep(5)
sinceEpochStart = time.time()
start_time = datetime.datetime.now()
endTime = start_time + datetime.timedelta(seconds=10)
markerTime = start_time + datetime.timedelta(milliseconds=500)
markerEndTime = markerTime + datetime.timedelta(seconds=3)
previous_buffer_count = 0

board.start_stream()

while True:
    # get buffer count
    buffer_count = board.get_board_data_count()
    
    # skip if no new packets in the buffer
    if previous_buffer_count == buffer_count:
        continue
    
    # get last MAX_PLOT_SIZE packets (those are not removed from the buffer, just copied)
    packets = board.get_current_board_data(MAX_PLOT_SIZE)
    
    # insert markers through a specific timeframe
    now = datetime.datetime.now()
    if markerTime <= now <= markerEndTime:
            board.insert_marker(buffer_count)

    # update the previous_buffer_count
    previous_buffer_count = buffer_count
    
    # update live plot
    # print(packets.transpose()[-1][eeg_channels[0]])
    update_plot(packets[eeg_channels + [timestamp_channel]].transpose())
    
    if datetime.datetime.now() >= endTime:
        break


board.stop_stream()

full_data = board.get_board_data()

board.release_session()

print("Streaming ended.")
plt.show()