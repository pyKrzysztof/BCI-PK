import sys
import json
import time
import queue
import random
import datetime
import importlib
import threading

from typing import Callable, List
from numpysocket import NumpySocket
from brainflow import BrainFlowInputParams, BoardShim, BoardIds, BrainFlowPresets


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


# Load pipeline parameters from JSON
def load_pipeline_parameters(json_filename: str):
    with open(json_filename, "r") as json_file:
        data = json.load(json_file)
        return data

def record_data(board_id, port, packet_size, session_timeout=3600, pipeline="", output_name="", electrode_config=None, simulated=False, server=False):

    # stop event and a data queue for server side.
    # start the server.
    if server:
        stop_event = threading.Event()
        data_queue = queue.Queue()
        start_server()
        

    # save the start time
    session_start_time = datetime.datetime.now().strftime("%d-%m-%Y-%H%M")
    
    # Set up board parameters
    params = BrainFlowInputParams()
    params.serial_port = port
    board = BoardShim(board_id if not simulated else BoardIds.SYNTHETIC_BOARD.value, params)

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
        if pipeline:
            # Load pipeline parameters from a JSON configuration file
            pipeline_parameters = load_pipeline_parameters(pipeline)

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
        raise
        print(e)
        print("Pipeline error. Exitting.")
        sys.exit()

    # Set up board and config
    board.prepare_session()
    if not simulated and electrode_config:
        for conf in electrode_config:
            print(board.config_board(conf))
            time.sleep(0.25)

    # Add the streamer output
    pipeline_name = pipeline[:-5] if pipeline != "" else "raw_data"
    file_name = f"{pipeline_name}_{session_start_time}.csv" if output_name == "" else output_name
    name = f"file://{file_name}:w"
    board.add_streamer(name, preset=BrainFlowPresets.DEFAULT_PRESET)

    # Start the stream
    board.start_stream()
    epoch_start = time.time()
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(seconds=session_timeout)
    previous_buffer_count = 0
    pipeline_end_flag = False
    packet_n = 0
    # Main loop
    condition = not stop_event.is_set() if server else True
    while condition:
        now = datetime.datetime.now()
        buffer_count = board.get_board_data_count()

        # Get the latest data from the board
        if buffer_count >= packet_size:
            packet_n = packet_n + buffer_count
            packets = board.get_board_data()
            if server:
                data = packets[send_channels]
                data_queue.put((data, epoch_start))
            # End the session one packet_size after pipeline ended (to not miss the last marker)
            if pipeline_end_flag:
                print("Finished the pipeline with", packet_n, "data packets.")
                processor.on_exit_callback()
                break

        # Ignore rest of the loop if no new data arrived.
        if buffer_count == previous_buffer_count:
            continue
        
        previous_buffer_count = buffer_count
        
        if pipeline:
            # Process the pipeline (if present)
            pipeline_end_flag = processor.process_pipeline(now)

        else:
            # End the session after a certain time if specified.
            if datetime.datetime.now() > end_time:
                if (session_timeout > 0):
                    break
                pass

    board.stop_stream()
    board.release_session()
    return file_name











# Function to send data from the queue in a separate thread
def server_thread(data_queue, connection: NumpySocket, server_stop_event):
    # Create filter parameters
    packet_count = 1
    try:
        while not server_stop_event.is_set() or not data_queue.empty():  # Check if the event is set and queue is not empty.
            try:
                # Get data from the queue to send (with a timeout to prevent indefinite blocking)
                data, epoch_start = data_queue.get(timeout=0.5)  # Adjust timeout as needed

                data[-1] = [timestamp - epoch_start for timestamp in data[-1]]
            
                # print(f"N: {packet_count}, Sending packets of shape: {data.shape}")
                connection.sendall(data)  # Send the data to the client
                packet_count = packet_count + 1
            except queue.Empty:
                continue  # If queue is empty, keep checking until signaled to stop
    # Handle exceptions due to a closed socket.
    except Exception as e:
        server_stop_event.set()
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


def start_server(ip, port, stop_event, data_queue, force_port=False):
    server_socket = NumpySocket()
    connected = False
    while True:
        try:
            server_socket.bind((ip, port))
            connected = True
        except:
            if not force_port:
                port = port - 1
            pass
        if connected:
            break

    server_socket.listen()

    print(f"Waiting for connection on port {port}...")
    connection, address = server_socket.accept()
    print("Client connected:", address)
    server_thread_instance = threading.Thread(target=server_thread, args=(data_queue, connection, stop_event, ))
    server_thread_instance.start()

    return server_socket, server_thread_instance

