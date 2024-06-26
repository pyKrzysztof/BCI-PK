import time
import argparse
import datetime
import queue
import threading
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from numpysocket import NumpySocket
import scipy.signal as signal
import random
from typing import Callable, List
import json
import importlib
import sys

# Function being executed with each action in a new thread. 
# A marker (action) is inserted as this function starts and when it returns (-action).
# This feature is now used with pipeline callbacks, see 'simple_pipeline.json' and 'comm_over_serial.py'
# def sample_action_function(action):
    # time.sleep(random.uniform(4, 6))


# Constants and session setup
session_start_time = datetime.datetime.now().strftime("%d-%m-%Y-%H%M%S")
ELECTRODES = 8
BOARD = BoardIds.CYTON_BOARD
SAMPLING_RATE = BoardShim.get_sampling_rate(BOARD.value)
PACKET_SIZE = 50
FILTER_NOTCH_FREQ = 50
FILTER_FACTOR = 30
BASE_THINKPULSE_CONFIG_GAIN_8 = [f"x{i+1}040010X" for i in range(ELECTRODES)]

# Create a queue to hold data between threads
data_queue = queue.Queue()

# Create server stop event
stop_event = threading.Event()

# Load pipeline parameters from JSON
def load_pipeline_parameters(json_filename: str):
    with open(json_filename, "r") as json_file:
        data = json.load(json_file)
        return data

# Represents a timetable entry with a random time delta and a specific action
class TimetableEntry:
    def __init__(self, delta_time, action):
        self.delta_time = delta_time
        self.action = action

# Generates a timetable with specified actions and random time deltas
class TimetableGenerator:
    def __init__(self, actions: List[str], min_time: float, max_time: float, n_samples: int):
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
        self.action_function()  # Simulate the action
        self.on_complete()  # Notify when done

# Process the pipeline and trigger actions based on current time
class PipelineProcessor:
    def __init__(self, timetable: List[TimetableEntry], action_identifiers: dict):
        self.timetable = timetable
        self.marker_insert_function = lambda *_: None
        self.action_identifiers = action_identifiers
        self.current_index = 0
        self.last_action_end_time = datetime.datetime.now()  # When the last action completed
        self.action_in_progress = False
        self.pending_markers = []  # Store pending markers for insertion in the main thread
    
    def process_pipeline(self, current_time: datetime.datetime):
        # Insert any pending markers from the main thread
        while self.pending_markers:
            marker = self.pending_markers.pop(0)  # Remove from front
            self.marker_insert_function(marker)
            # print(f"Inserting marker {marker}")

        # Calculate the time since the last action ended
        time_since_last_action = (current_time - self.last_action_end_time).total_seconds()

        if not self.action_in_progress:
            if self.current_index < len(self.timetable):
                time_to_next_action = self.timetable[self.current_index].delta_time

                # If enough time has passed, start the next action
                if time_since_last_action >= time_to_next_action:
                    self.start_next_action()
            else:
                return 1 

    def start_next_action(self):
        entry = self.timetable[self.current_index]
        action_identifier = self.action_identifiers[entry.action]  # Unique identifier for the action
        
        # Queue the start marker for the main thread to insert
        self.pending_markers.append(action_identifier)

        # Define the completion callback
        def on_action_complete():
            # Queue the end marker for the main thread to insert
            # print(f"Finishing action {action_identifier}")
            self.pending_markers.append(-action_identifier)
            self.action_in_progress = False
            # Update the last action's end time
            self.last_action_end_time = datetime.datetime.now()
        
        # Create the action executor and start it
        action_executor = ActionExecutor(
            entry.action,
            lambda: self.action_callback_func(entry.action),
            on_action_complete
        )
        action_executor.start()  # Start non-blocking action
        self.action_in_progress = True
        self.current_index += 1
        
    def set_marker_function(self, func):
        self.marker_insert_function = func


# Function to send data from the queue in a separate thread
def server_thread(args, connection: NumpySocket, server_stop_event):
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
                    for i in range(ELECTRODES):
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
        if not args.nostreamer:
            print("Session will continue as there's an active backup streamer.")
        else:
            server_stop_event.set()
            print("Session won't continue as there is no active backup streamer. Consider adding '-streamer' as a parameter.")
    finally:
        # 2 Disables a Socket for both sending and receiving. This field is constant.
        # 1 Disables a Socket for sending. This field is constant.
        # 0 Disables a Socket for receiving. This field is constant.
        try:
            connection.shutdown(2)
        except:
            pass
        finally:
            connection.close()

# Function to gather data in the main thread
def gather_data(args):
    
    # Set up board parameters
    params = BrainFlowInputParams()
    params.serial_port = args.device_port
    board = BoardShim(BOARD.value if not args.simulated else BoardIds.SYNTHETIC_BOARD.value, params)

    # Set up board channel info
    package_num_channel = BoardShim.get_package_num_channel(board.board_id)
    eeg_channels = BoardShim.get_eeg_channels(board.board_id)
    marker_channel = BoardShim.get_marker_channel(board.board_id)
    # gyro_channels = BoardShim.get_gyro_channels(board.board_id)
    timestamp_channel = BoardShim.get_timestamp_channel(board.board_id)
    
    # Select channels to be sent over TCP/IP
    send_channels = eeg_channels
    # send_channels.extend(gyro_channels)
    send_channels.append(marker_channel)
    send_channels.append(timestamp_channel)
    
    # Prepare pipeline (if provided)
    try:
        if args.pipeline:
            # Load pipeline parameters from a JSON configuration file
            pipeline_parameters = load_pipeline_parameters(args.pipeline)

            # Extract parameters from JSON data
            actions = pipeline_parameters["actions"]
            action_identifiers = {name: idx+1 for idx, name in enumerate(actions)}  # Unique identifiers for each action

            min_time = pipeline_parameters["time_between_actions_min"]
            max_time = pipeline_parameters["time_between_actions_max"]
            n_samples = pipeline_parameters["n_samples"]

            # Generate timetable
            generator = TimetableGenerator(actions, min_time, max_time, n_samples)
            timetable = generator.generate_timetable()

            # Create the pipeline processor with action identifiers
            processor = PipelineProcessor(timetable, action_identifiers)
            processor.set_marker_function(board.insert_marker)

            # Prepare the pipeline callbacks
            init_module, init_func = pipeline_parameters["call_on_enter"].rsplit(".", 1)
            action_module, action_func = pipeline_parameters["call_on_action"].rsplit(".", 1)
            close_module, close_func = pipeline_parameters["call_on_exit"].rsplit(".", 1)
            assert init_module == action_module == close_module
            assert init_func != action_func
            module = importlib.import_module(init_module)
            init_func_handle = getattr(module, init_func)
            action_func_handle = getattr(module, action_func)
            close_func_handle = getattr(module, close_func)

            init_params = init_func_handle()
            processor.action_callback_func = lambda action: action_func_handle(init_params, action)
            processor.on_exit_callback = lambda: close_func_handle(init_params)

    except Exception as e:
        print(e)
        print("Pipeline error. Exitting.")
        sys.exit()

    # Set up board and config
    board.prepare_session()
    if not args.simulated and args.config:
        for conf in BASE_THINKPULSE_CONFIG_GAIN_8:
            print(board.config_board(conf))
            time.sleep(0.25)

    # Add the streamer output
    if not args.nostreamer:
        pipeline_name = args.pipeline[:-5] if args.pipeline != "" else "raw_data"
        name = f"file://{pipeline_name}_{session_start_time}.csv:w"
        board.add_streamer(name, preset=BrainFlowPresets.DEFAULT_PRESET)

    # Start the stream
    board.start_stream()
    epoch_start = time.time()
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(seconds=args.time)
    previous_buffer_count = 0
    pipeline_end_flag = False
    packet_n = 0
    # Main loop
    while not stop_event.is_set():
        now = datetime.datetime.now()
        buffer_count = board.get_board_data_count()

        # Get the latest data from the board
        if buffer_count >= PACKET_SIZE:
            packet_n = packet_n + buffer_count
            packets = board.get_board_data()
            data = packets[send_channels]
            data_queue.put((data, epoch_start))
            # End the session one packet_size after pipeline ended (to not miss the last marker)
            if pipeline_end_flag:
                print("Finished the pipeline. ", packet_n)
                processor.on_exit_callback()
                break

        # Ignore rest of the loop if no new data arrived.
        if buffer_count == previous_buffer_count:
            continue
        
        previous_buffer_count = buffer_count
        
        if args.pipeline:
            # Process the pipeline (if present)
            pipeline_end_flag = processor.process_pipeline(now)

        else:
            # End the session after a certain time if specified.
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
            if not parsed_args.force_port:
                port = port - 1
            pass
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
    parser.add_argument("--device_port", type=str, default="/dev/ttyUSB0", required=False)
    parser.add_argument("--time", type=int, default=0, required=False, help="Time of the simulation, will run indefinitely if not provided or as long as it needs when --pipeline is passed.")
    parser.add_argument("--pipeline", type=str, default="", required=False, help="Pass a .json file with pipeline configuration to process the pipeline, overwrites --time parameter.")
    parser.add_argument("-nostreamer", action="store_true", help="Prevents the streaming of board data into a .csv file with current date and time as the filename.")
    parser.add_argument("-filter", action="store_true", help="Enables a notch filter at 50Hz")
    parser.add_argument("-simulated", action="store_true", help="Simulates a board if no physical connection is present.")
    parser.add_argument("-server", action="store_true", help="Runs the socket server.")
    parser.add_argument("-markers", action="store_true", help="Decides if the marker channel will be passed as second to last row of data over the socket.")
    parser.add_argument("-config", action="store_true", help="Send active electrode config.")
    parser.add_argument("-force_port", action="store_true", required=False, default=False)
    args = parser.parse_args()
    print("Running a session with:", args)
    
    # Set up the server socket and start the server thread
    if args.server:
        server_socket, server_thread_instance = start_server(args)

    # Run the data-gathering function in the main thread
    try:
        gather_data(args)
    except Exception as e:
        raise e
    finally:
        if args.server:
            stop_event.set()
            server_socket.close()  # Close the server when done
            server_thread_instance.join()  # Wait for the server thread to complete


if __name__ == "__main__":
    main()

