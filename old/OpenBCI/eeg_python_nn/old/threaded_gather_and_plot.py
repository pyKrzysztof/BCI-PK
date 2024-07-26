import threading
import queue
import time
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter


# Session start time
session_start_time = datetime.datetime.now().strftime("%d-%m-%Y-%H%M%S")

# Constants
SIMULATED = True 
ELECTRODES = range(1, 9)
BOARD = BoardIds.CYTON_BOARD
PORT = "/dev/ttyS3"
MAX_PLOT_SIZE = 500

# Configuration presets
BASE_THINKPULSE_CONFIG_GAIN_1 = [f"x{i}000010X" for i in ELECTRODES]
BASE_THINKPULSE_CONFIG_GAIN_2 = [f"x{i}010010X" for i in ELECTRODES]
BASE_THINKPULSE_CONFIG_GAIN_4 = [f"x{i}020010X" for i in ELECTRODES]
BASE_THINKPULSE_CONFIG_GAIN_6 = [f"x{i}030010X" for i in ELECTRODES]
BASE_THINKPULSE_CONFIG_GAIN_8 = [f"x{i}040010X" for i in ELECTRODES]

# Queue to communicate between threads
data_queue = queue.Queue()

sinceEpochStart = time.time()

# Function to gather data in a separate thread
def gather_data():
    global sinceEpochStart
    # Elevate process priority (need superuser permissions)
    os.nice(-10)
    
    # Set up board parameters
    params = BrainFlowInputParams()
    params.serial_port = PORT
    board = BoardShim(BOARD.value if not SIMULATED else BoardIds.SYNTHETIC_BOARD.value, params)
    
    # Set up board channel info
    package_num_channel = BoardShim.get_package_num_channel(board.board_id)
    eeg_channels = BoardShim.get_eeg_channels(board.board_id)
    marker_channel = BoardShim.get_marker_channel(board.board_id)
    timestamp_channel = BoardShim.get_timestamp_channel(board.board_id)
    refresh_rate = 1 / BoardShim.get_sampling_rate(BOARD.value)

    # Setup session and config
    board.prepare_session()
    config = "".join(BASE_THINKPULSE_CONFIG_GAIN_8)
    board.config_board(config)
    board.add_streamer(f"file://{session_start_time}_default.csv:w", preset=BrainFlowPresets.DEFAULT_PRESET)
    
    # Start stream
    board.start_stream()
    sinceEpochStart = time.time()
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(seconds=10)
    marker_time = start_time + datetime.timedelta(milliseconds=500)
    marker_endtime = marker_time + datetime.timedelta(seconds=3)
    previous_buffer_count = 0

    while True:
        now = datetime.datetime.now()
        buffer_count = board.get_board_data_count()

        if previous_buffer_count == buffer_count:
            continue

        # Get the latest data from the board
        packets = board.get_current_board_data(MAX_PLOT_SIZE)
        data_queue.put(packets[eeg_channels + [timestamp_channel]].transpose())
        previous_buffer_count = buffer_count

        if marker_time <= now <= marker_endtime:
            board.insert_marker(buffer_count)

        if datetime.datetime.now() > end_time:
            break
        
    board.stop_stream()
    board.release_session()

# Start the data-gathering thread
thread = threading.Thread(target=gather_data)
thread.start()

# Set up the plot
fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(4, 8), constrained_layout=True)

# Function to update the plot
def update_plot(packets):
    relative_times = [channels[-1] - sinceEpochStart for channels in packets]

    for i, ax in enumerate(axes):
        ax.clear()
        ax.plot(relative_times,  [channel[i] for channel in packets])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    fig.canvas.draw()
    plt.pause(0.004)

# Main loop to update the plot in real-time
while thread.is_alive():
    pass
    try:
        # Try to get the latest data from the queue
        packets = data_queue.get_nowait()
        update_plot(packets)
    except queue.Empty:
        # If the queue is empty, just wait a bit and continue
        time.sleep(0.001)
        pass

plt.show()  # Hold the final plot