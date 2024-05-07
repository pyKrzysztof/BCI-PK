import time
import os
import argparse
import datetime
import numpy as np
import queue
import threading
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter
from numpysocket import NumpySocket

# Constants and session setup
session_start_time = datetime.datetime.now().strftime("%d-%m-%Y-%H%M%S")
SIMULATED = True
ELECTRODES = range(1, 9)
BOARD = BoardIds.CYTON_BOARD
PACKET_SIZE = 25

# Configuration presets
BASE_THINKPULSE_CONFIG_GAIN_8 = [f"x{i}040010X" for i in ELECTRODES]

# Create a queue to hold data between threads
data_queue = queue.Queue()

# Create server stop event
server_stop_event = threading.Event()

# Function to read and create a pipeline function from a file.
def load_pipeline(path):
    return lambda *_: None

# Function to send data from the queue in a separate thread
def server_thread(args, connection, stop_event):
    packet_count = 1
    try:
        while not stop_event.is_set() or not data_queue.empty():  # Check if the event is set and queue is not empty.
            try:
                # Get data from the queue to send (with a timeout to prevent indefinite blocking)
                data, epoch_start = data_queue.get(timeout=0.5)  # Adjust timeout as needed
                for row in data:
                    row[-1] = row[-1] - epoch_start
                print(f"N: {packet_count}, Sending packets of shape: {data.shape}")
                connection.sendall(data)  # Send the data to the client
                packet_count = packet_count + 1
            except queue.Empty:
                continue  # If queue is empty, keep checking until signaled to stop
    except Exception as e:
        print("Server error:", e, "\n Session will continue but no packets will be sent. Consider adding -streamer flag to not lose session data.")
    finally:
        connection.close()

# Function to gather data in the main thread
def gather_data(args):
    # Set up board parameters
    params = BrainFlowInputParams()
    params.serial_port = args.device_port
    board = BoardShim(BOARD.value if not SIMULATED else BoardIds.SYNTHETIC_BOARD.value, params)

    # Set up board channel info
    package_num_channel = BoardShim.get_package_num_channel(board.board_id)
    eeg_channels = BoardShim.get_eeg_channels(board.board_id)
    marker_channel = BoardShim.get_marker_channel(board.board_id)
    timestamp_channel = BoardShim.get_timestamp_channel(board.board_id)
    sampling_rate = BoardShim.get_sampling_rate(BOARD.value)
    
    # Prepare pipeline (if provided)
    if args.pipeline is not None:
        process_pipeline = load_pipeline(args.pipeline)
    else:
        process_pipeline = lambda *_: None

    # Set up board and config
    board.prepare_session()
    config = "".join(BASE_THINKPULSE_CONFIG_GAIN_8)
    board.config_board(config)
    
    # Add the streamer output
    if args.streamer:
        board.add_streamer(f"file://{session_start_time}_default.csv:w", preset=BrainFlowPresets.DEFAULT_PRESET)
    
    # Start the stream
    board.start_stream()
    epoch_start = time.time()
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(seconds=args.time)
    previous_buffer_count = 0
    
    # Main loop
    while True:
        
        now = datetime.datetime.now()
        buffer_count = board.get_board_data_count()

        # Get the latest data from the board
        if buffer_count >= PACKET_SIZE:
            packets = board.get_board_data()
            data = packets[eeg_channels + [timestamp_channel]].transpose()
            data_queue.put((data, epoch_start))

        if buffer_count == previous_buffer_count:
            continue

        process_pipeline(board, now, start_time)
        previous_buffer_count = buffer_count

        if datetime.datetime.now() > end_time:
            break

    board.stop_stream()
    board.release_session()

# Function to start the server and run it's operation in a new thread after establishing connection
def start_server(parsed_args):
    server_socket = NumpySocket()
    port = parsed_args.port
    connected = False
    while True:
        try:
            server_socket.bind((parsed_args.ip, port))
            connected = True
        except:
            port = port - 1
        if connected:
            break

    server_socket.listen()

    print(f"Waiting for connection on port {port}...")
    connection, address = server_socket.accept()
    print("Client connected:", address)
    server_thread_instance = threading.Thread(target=server_thread, args=(parsed_args, connection, server_stop_event, ))
    server_thread_instance.start()

    return server_socket, server_thread_instance



def main():
    # Parser setup
    parser = argparse.ArgumentParser("threaded_gathering_socket")
    parser.add_argument("--ip", type=str, default="localhost", required=False)
    parser.add_argument("--port", type=int, default=9999, required=False)
    parser.add_argument("--device_port", type=str, default="/dev/ttyS3", required=False)
    parser.add_argument("--time", type=int, default=10, required=True)
    parser.add_argument("--pipeline", type=str, default='', required=False)
    parser.add_argument("-streamer", action="store_true")
    parser.add_argument("-filter", action="store_true")
    args = parser.parse_args()
    print("Running a session with:", args)

    # Set up the server socket and start the server thread
    server_socket, server_thread_instance = start_server(args)

    # Run the data-gathering function in the main thread
    try:
        gather_data(args)
    except Exception as e:
        raise e
    finally:
        server_stop_event.set()
        server_socket.close()  # Close the server when done
        server_thread_instance.join()  # Wait for the server thread to complete



if __name__ == "__main__":
    main()