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
from numpysocket import NumpySocket
import scipy.signal as signal
import random
from typing import Callable


# Represents a timetable entry with a random time delta and a specific action
class TimetableEntry:
    def __init__(self, delta_time, action):
        self.delta_time = delta_time
        self.action = action

# Generates a timetable with specified actions and random time deltas
class TimetableGenerator:
    def __init__(self, actions: list[str], min_time: float, max_time: float, n_samples: int):
        self.actions = actions
        self.min_time = min_time
        self.max_time = max_time
        self.n_samples = n_samples

    def generate_timetable(self):
        timetable = []
        for action in self.actions:
            for _ in range(self.n_samples):
                delta_time = random.uniform(self.min_time, self.max_time)
                timetable.append(TimetableEntry(delta_time, action))
        random.shuffle(timetable)  # Shuffle to ensure randomness in sequence
        return timetable

# Non-blocking action executor with completion callback
class ActionExecutor(threading.Thread):
    def __init__(self, action_name, action_function, on_complete: Callable):
        threading.Thread.__init__(self)
        self.action_name = action_name
        self.action_function = action_function
        self.on_complete = on_complete

    def run(self):
        self.action_function()
        self.on_complete()  # Notify when done

# Process the pipeline and trigger actions sequentially with relative time deltas
class PipelineProcessor:
    def __init__(self, timetable: list[TimetableEntry], action_identifiers: dict):
        self.timetable = timetable
        self.marker_insert_function = lambda *_: None
        self.action_identifiers = action_identifiers
        self.current_index = 0
        self.time_to_next_action = 0  # Time remaining until the next action can start
        self.action_in_progress = False
        self.pending_markers = []  # Store pending markers for insertion in the main thread
    
    def process_pipeline(self, time_delta: float):
        # Insert any pending markers from the main thread
        while self.pending_markers:
            marker = self.pending_markers.pop(0)  # Remove from front
            self.marker_insert_function(marker)
            print(f"Inserting marker {marker}")

        self.time_to_next_action -= time_delta

        if self.time_to_next_action <= 0 and not self.action_in_progress and self.current_index < len(self.timetable):
            self.start_next_action()

        if self.current_index == len(self.timetable):
            return 1
        

    def start_next_action(self):
        entry = self.timetable[self.current_index]
        action_identifier = self.action_identifiers[entry.action]  # Unique identifier for the action
        
        # Queue the start marker for the main thread to insert
        self.pending_markers.append(action_identifier)

        # Define the completion callback
        def on_action_complete():
            # Queue the end marker for the main thread to insert
            self.pending_markers.append(-action_identifier)
            self.action_in_progress = False
        
        # Create the action executor and start it
        action_executor = ActionExecutor(
            entry.action,
            lambda: time.sleep(5),  # Simulate the action
            on_action_complete
        )
        print("Starting action..")
        action_executor.start()  # Start non-blocking action
        self.action_in_progress = True
        self.current_index += 1

        if self.current_index < len(self.timetable):
            # Set time to the next action based on the current timetable entry
            self.time_to_next_action = self.timetable[self.current_index].delta_time
            print("Time to next action:", self.time_to_next_action)

    def set_marker_function(self, func):
        self.marker_insert_function = func






# Constants and session setup
session_start_time = datetime.datetime.now().strftime("%d-%m-%Y-%H%M%S")
ELECTRODES = range(1, 9)
BOARD = BoardIds.CYTON_BOARD
PACKET_SIZE = 25
FILTER_NOTCH_FREQ = 50
FILTER_FACTOR = 30
SAMPLING_RATE = BoardShim.get_sampling_rate(BOARD.value)

# Configuration presets
BASE_THINKPULSE_CONFIG_GAIN_8 = [f"x{i}040010X" for i in ELECTRODES]

# Create a queue to hold data between threads
data_queue = queue.Queue()

# Create server stop event
stop_event = threading.Event()

# Function to send data from the queue in a separate thread
def server_thread(args, connection, server_stop_event):
    
    # Create filter parameters
    if args.filter:
        b, a = signal.iirnotch(FILTER_NOTCH_FREQ, FILTER_FACTOR, fs=SAMPLING_RATE)

    packet_count = 1
    try:
        while not server_stop_event.is_set() or not data_queue.empty():  # Check if the event is set and queue is not empty.
            try:
                # Get data from the queue to send (with a timeout to prevent indefinite blocking)
                data, epoch_start = data_queue.get(timeout=0.5)  # Adjust timeout as needed
                
                if args.filter:
                    for i in range(8):
                        data[i] = signal.lfilter(b, a, data[i])

                data[-1] = [timestamp - epoch_start for timestamp in data[-1]]
                
                # print(f"N: {packet_count}, Sending packets of shape: {data.shape}")
                connection.sendall(data)  # Send the data to the client
                packet_count = packet_count + 1
            except queue.Empty:
                continue  # If queue is empty, keep checking until signaled to stop

    # Handle exceptions due to a closed socket.
    except Exception as e:
        print("Server crashed.")
        if args.streamer:
            print("Session will continue as there's an active backup streamer.")
        else:
            server_stop_event.set()
            print("Session won't continue as there is no active backup streamer. Consider adding '-streamer' as a parameter.")
    finally:
        connection.close()

# Function to gather data in the main thread
def gather_data(args, processor : PipelineProcessor):
    # Set up board parameters
    params = BrainFlowInputParams()
    params.serial_port = args.device_port
    board = BoardShim(BOARD.value if not args.simulated else BoardIds.SYNTHETIC_BOARD.value, params)

    # Set up board channel info
    package_num_channel = BoardShim.get_package_num_channel(board.board_id)
    eeg_channels = BoardShim.get_eeg_channels(board.board_id)
    marker_channel = BoardShim.get_marker_channel(board.board_id)
    timestamp_channel = BoardShim.get_timestamp_channel(board.board_id)
    
    # Prepare pipeline (if provided)
    if args.pipeline is not None:
        processor.set_marker_function(board.insert_marker)

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
    prev_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(seconds=args.time)
    previous_buffer_count = 0
    
    send_channels = eeg_channels
    if args.markers:
        send_channels.append(marker_channel)
    send_channels.append(timestamp_channel)
    # Main loop
    while not stop_event.is_set():
        
        now = datetime.datetime.now()
        buffer_count = board.get_board_data_count()

        # Get the latest data from the board
        if buffer_count >= PACKET_SIZE:
            packets = board.get_board_data()
            data = packets[send_channels]
            data_queue.put((data, epoch_start))

        if buffer_count == previous_buffer_count:
            continue
        
        if processor.process_pipeline((now - prev_time).total_seconds()):
            break

        previous_buffer_count = buffer_count

        if datetime.datetime.now() > end_time:
            if (args.time > 0):
                break
            pass

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
    server_thread_instance = threading.Thread(target=server_thread, args=(parsed_args, connection, stop_event, ))
    server_thread_instance.start()

    return server_socket, server_thread_instance



def main():
    # Parser setup
    parser = argparse.ArgumentParser("threaded_gathering_socket")
    parser.add_argument("--ip", type=str, default="localhost", required=False)
    parser.add_argument("--port", type=int, default=9999, required=False)
    parser.add_argument("--device_port", type=str, default="/dev/ttyS3", required=False)
    parser.add_argument("--time", type=int, default=0, required=False)
    # parser.add_argument("--pipeline", type=str, default='', required=False)
    parser.add_argument("-pipeline", action="store_true")
    parser.add_argument("-streamer", action="store_true")
    parser.add_argument("-filter", action="store_true")
    parser.add_argument("-simulated", action="store_true")
    parser.add_argument("-noserver", action="store_true")
    parser.add_argument("-markers", action="store_true", help="Decides if the marker channel will be passed as second to last row of data over the socket.")
    args = parser.parse_args()
    print("Running a session with:", args)
    
    # Define pipeline parameters
    actions = ["LEFT", "RIGHT"]
    action_identifiers = {name: idx+1 for idx, name in enumerate(actions)}
    min_time = 3
    max_time = 4
    n_samples = 2

    # Generate timetable
    generator = TimetableGenerator(actions, min_time, max_time, n_samples)
    timetable = generator.generate_timetable()

    # Create the pipeline processor
    processor = PipelineProcessor(timetable, action_identifiers)

    # Set up the server socket and start the server thread
    if not args.noserver:
        server_socket, server_thread_instance = start_server(args)

    # Run the data-gathering function in the main thread
    try:
        gather_data(args, processor)
    except Exception as e:
        raise e
    finally:
        if not args.noserver:
            stop_event.set()
            server_socket.close()  # Close the server when done
            server_thread_instance.join()  # Wait for the server thread to complete



if __name__ == "__main__":
    main()