import time
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter
from numpysocket import NumpySocket

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




# Function to gather data
def gather_data(connection : NumpySocket):
    # Elevate process priority (need superuser permissions)
    # os.nice(-10)
    
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
        
        # print(packets.shape)
        data = packets[eeg_channels + [timestamp_channel]].transpose()
        print(connection.sendall(data))
        
        previous_buffer_count = buffer_count

        if marker_time <= now <= marker_endtime:
            board.insert_marker(buffer_count)

        if datetime.datetime.now() > end_time:
            break

        
    board.stop_stream()
    board.release_session()

with NumpySocket() as socket:
    # Set up a server socket
    socket.bind(("localhost", 9999))
    socket.listen()
    
    print("Waiting for connection.")
    connection, address = socket.accept()
    print("Graph connected:", address)

    try:
        gather_data(connection)
    except Exception as e:
        print(e)
    finally:
        socket.close()
    
