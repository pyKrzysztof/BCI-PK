import time
import math
import datetime
import collections
import numpy as np
import dearpygui.dearpygui as dpg

from queue import Queue, Empty
from multiprocessing.synchronize import Event
from brainflow import BoardIds, BoardShim, BrainFlowInputParams

from mybci.system import System, SystemConfig, Statistics, Filesystem, DataReader
from mybci.feature_extraction import get_filter_function


def file_open_callback(sender, app_data):
    print('OK was clicked.')
    print("Sender: ", sender)
    print("App Data: ", app_data)


def toggle_button_active(_, state, tag):
    if state:
        dpg.enable_item(tag)
    else:
        dpg.disable_item(tag)


class SystemGUIManager:


    gui_running = False
    system: System = None
    queues: dict[str, Queue] = None
    queue_raw: Queue = None
    queue_processed: Queue = None
    queue_cmd_tx: Queue = None
    queue_cmd_rx: Queue = None
    end_event: Event = None

    use_reinforcement = True

    live_eeg_data = []
    live_eeg_autofit = False


    def __init__(self, system: System, data: dict):
        self.system = system

        # self.queues: dict[str, Queue] = {'raw': data['raw'], 'processed': data['processed'], 'cmd_tx': data['cmd_tx'], 'cmd_rx': data['cmd_rx']}
        self.queue_raw: Queue = data['raw']
        self.queue_processed: Queue = data['processed']
        self.queue_cmd_tx: Queue = data['cmd_tx']
        self.queue_cmd_rx: Queue = data['cmd_rx']
        self.end_event: Event = data['event_end']

        self.live_eeg_data = [collections.deque(maxlen=500) for _ in range(len(self.system.config.channels_eeg))]
        
    def create_eeg_window(self):

        with dpg.window(label='EEG', tag='eegwin', pos=(1100, 20), width=480, height=-1, collapsed=True):
            colors = [
                (244, 164, 96, 255),  # Sandy Brown
                (176, 196, 222, 255), # Light Steel Blue
                (255, 218, 185, 255), # Peach Puff
                (143, 188, 143, 255), # Dark Sea Green
                (250, 128, 114, 255), # Salmon
                (221, 160, 221, 255), # Plum
                (60, 179, 113, 255),  # Medium Sea Green
                (216, 191, 216, 255)  # Thistle
            ]
            
            electrode_indices = [j+1 for j in range(len(self.system.config.channels_eeg))]

            def set_axis_limits(auto):
                if auto:
                    self.live_eeg_autofit = True
                else:
                    self.live_eeg_autofit = False

                for i in electrode_indices:
                    if self.live_eeg_autofit:
                        dpg.set_axis_limits_auto(f"eeg-y{i}")
                    else:    
                        ys = dpg.get_value("eeg-y-limits")
                        dpg.set_axis_limits(f"eeg-y{i}", ys[0], ys[1])
            

            dpg.add_input_intx(label="Y Limits", size=2, default_value=[-400, 400], tag="eeg-y-limits", callback= lambda: set_axis_limits(dpg.get_value("auto-limits-checkbox")))
            dpg.add_checkbox(label="Auto limits", tag="auto-limits-checkbox" ,callback=lambda _, state: set_axis_limits(state), default_value=False)


            for i in electrode_indices:

                with dpg.theme(tag=f"plot_theme_{i}"):
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, colors[i-1], category=dpg.mvThemeCat_Plots)
                
                with dpg.plot(label='', height=80, width=-1, no_title=True, no_mouse_pos=True, no_menus=True, no_box_select=True, no_highlight=True):

                    # REQUIRED: create x and y axes, set to auto scale.
                    x_axis = dpg.add_plot_axis(dpg.mvXAxis, tag=f'eeg-x{i}', no_tick_labels=True, no_gridlines=True, no_tick_marks=True)
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, tag=f'eeg-y{i}', no_tick_labels=True, no_gridlines=True, no_tick_marks=True)
                    # series belong to a y axis.
                    dpg.add_line_series(x=list(), y=list(), label='', parent=f'eeg-y{i}', tag=f'eeg-series{i}')

                    dpg.bind_item_theme(f"eeg-series{i}", f"plot_theme_{i}")

            set_axis_limits(0)

    def update_eeg_window(self, data, count):
        x_data = [i for i in range(count-len(data[0]), count)]

        for i in range(1, 9):
            dpg.set_value(f'eeg-series{i}', [x_data, list(data[i-1])])
            dpg.fit_axis_data(f'eeg-x{i}')
            if self.live_eeg_autofit:
                dpg.fit_axis_data(f'eeg-y{i}')

    def create_system_windows(self):
        """Context must be created before calling this function."""

        with dpg.window(label="System Panel", no_close=True, pos=(20, 20), width=400, height=500, no_move=True, no_resize=True, collapsed=False):

            # Save system header
            with dpg.collapsing_header(label="Save System"):

                dpg.add_button(label="Save System", callback=self.system.save_system, width=-1)

            # System information header
            with dpg.collapsing_header(label="System Information"):

                dpg.add_text(f"Actions: {self.system.config.actions}", tag="info_actions")

            # EEG Board Header
            with dpg.collapsing_header(label="EEG Board", default_open=True):

                # Device Header
                with dpg.collapsing_header(label="Device", default_open=True, indent=20):

                    dpg.add_input_text(label="Board Port", default_value=self.system.config.board_port, tag="board_port", 
                                       callback=lambda _, value: setattr(self.system.config, "board_port", value))
                    
                    dpg.add_listbox([self.system.config.board_type, BoardIds.SYNTHETIC_BOARD], tag="board-id-selector")

                    dpg.add_button(label="Connect", width=-1, tag="board-connect-button", 
                                   callback=lambda: self.send_cmd(["connect", {"board_port": self.system.config.board_port, "board_id": dpg.get_value("board-id-selector")}]))
                    
                    dpg.add_button(label="Start Stream", width=-1, tag="board-start-button", 
                                   callback=lambda: self.send_cmd(["start_stream", {"session": "standard", "send_raw": True, "send_processed": True, "send_features": True, "config": self.system.config}]))
                    
                    dpg.add_button(label="Stop Stream", width=-1, tag="board-stop-button", 
                                   callback=lambda: self.send_cmd("stop_stream"))

                    dpg.add_button(label="Disconnect", width=-1, tag="board-disconnect-button", 
                                   callback=lambda: self.send_cmd("disconnect"))

                # Training Header
                with dpg.collapsing_header(label="Training", tag="training-header", indent=20):
                    
                    dpg.add_checkbox(label="Use pipeline", callback= lambda _, state: toggle_button_active(_, state, "pipeline_select_button"))

                    with dpg.file_dialog(directory_selector=False, show=False, tag="pipeline_dialog", width=700, height=400,
                                         callback=file_open_callback):
                        dpg.add_file_extension(".json", color=(20, 180, 180, 255), custom_text="[JSON]")

                    dpg.add_button(label="Select Pipeline", width=-1, tag="pipeline_select_button", enabled=False,
                                   callback=lambda: dpg.show_item("pipeline_dialog"))

                    with dpg.file_dialog(directory_selector=False, show=False, tag="model_dialog", width=700, height=400,
                                         callback=file_open_callback):
                        dpg.add_file_extension(".py", color=(20, 100, 180, 255), custom_text="[PYTHON]")
                        dpg.add_file_extension(".keras", color=(20, 180, 180, 255), custom_text="[KERAS]")

                    dpg.add_button(label="Select Model", width=-1, tag="model_select_button", enabled=True,
                                   callback=lambda: dpg.show_item("model_dialog"))

                    # Reinforcement Header
                    with dpg.collapsing_header(label="Reinforcement", parent="training-header", indent=40):
                        # with dpg.file_dialog(directory_selector=False, show=False, callback=file_open_callback,  tag="reinforcement_dialog", width=700 ,height=400):
                        #     dpg.add_file_extension(".py", color=(20, 80, 180, 255), custom_text="[PYTHON]")

                        dpg.add_checkbox(label="Use reinforcement.", default_value=True,
                                         callback= lambda _, state: setattr(self, "use_reinforcement", state))
                        # dpg.add_button(label="Select Reinforcement App", width=-1, callback=lambda: dpg.show_item("reinforcement_dialog"), tag="reinforcement_select_button", enabled=False)

        # EEG Window
        self.create_eeg_window()

    def create_gui(self, custom_gui_handle:callable = None, kwargs:dict = None):
        
        # create context and viewport
        dpg.create_context()
        # dpg.configure_app(init_file="custom_layout.ini")
        dpg.create_viewport(title="System", width=1600, height=900)
        dpg.setup_dearpygui()

        # create base system windows
        self.create_system_windows()

        # load custom gui for the application
        if custom_gui_handle is not None:
            custom_gui_handle(self, kwargs)
        
        dpg.show_viewport()

    def mainloop_gui(self, custom_mainloop_handle:callable = None, kwargs:dict = None):
        self.gui_running = True
        raw_data_count = 0

        while dpg.is_dearpygui_running() and not self.end_event.is_set():

            queue_updates = self.check_queues()

            if queue_updates["raw"] is not None:
                board_data = queue_updates["raw"].transpose()
                raw_data_count += board_data.shape[1]
                for i, channel in enumerate(self.system.config.channels_eeg):
                    self.live_eeg_data[i].extend(board_data[channel])
                self.update_eeg_window(self.live_eeg_data, raw_data_count)

            if custom_mainloop_handle is not None:
                custom_mainloop_handle(self, kwargs)
            
            dpg.render_dearpygui_frame()

        dpg.destroy_context()
        self.end_event.set()

    def send_cmd(self, cmd):
        self.queue_cmd_tx.put_nowait(cmd)

    def check_queues(self):
        cmd = self._check_queue(self.queue_cmd_rx)
        raw_data = self._check_queue(self.queue_raw)
        processed_data = self._check_queue(self.queue_processed)
        data = {"cmd": cmd, "raw": raw_data, "processed": processed_data}
        return data

    def _check_queue(self, queue):
        try:
            return queue.get_nowait()
        except:
            return None












class DataProcessor:


    queue_raw: Queue = None
    queue_processed: Queue = None
    queue_cmd_tx: Queue = None
    queue_cmd_rx: Queue = None

    board_handle: BoardShim = None
    board_params: BrainFlowInputParams = None
    board_connected = False
    board_streaming = False

    stream_reader: DataReader = None
    stream_runner: callable = lambda: None


    def __init__(self, data):
        self.queue_raw: Queue = data['raw']
        self.queue_processed: Queue = data['processed']
        self.queue_cmd_tx: Queue = data['cmd_rx']
        self.queue_cmd_rx: Queue = data['cmd_tx']
        self.end_event: Event = data['event_end']


    def update(self):
        self.process_commands()

        if self.board_streaming:
            self.stream_runner()


        if self.end_event.is_set():
            if self.board_connected:
                self.disconnect()
            return 0

        return 1

    def connect(self, params):
        self.board_params = BrainFlowInputParams()
        try:
            params["board_id"] = eval(params["board_id"])
        except:
            pass
        print(params)
        if params["board_id"] != BoardIds.SYNTHETIC_BOARD:
            self.board_params.serial_port = params['board_port']

        try:
            self.board_handle = BoardShim(params['board_id'], self.board_params)
            self.board_connected = True
        except:
            try:
                self.queue_cmd_tx.put_nowait(["Error", "Failed to connect to the board."])
            except:
                pass

    def start_stream(self, params):
        if not self.board_connected:
            return
        
        self.board_handle.release_all_sessions()
        self.board_handle.prepare_session()

        if not self.board_handle.is_prepared():
            return

        name = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
        self.board_handle.add_streamer(f"file://temp/raw_{name}.csv:w")

        stream_type = params["session"]
        config:SystemConfig = params["config"]
        overlap = False

        if stream_type == "standard":
            overlap = False
            self.stream_reader = DataReader(self.board_handle, config.packet_size, config.feature_size, config.separator)
            self.stream_runner = lambda: self.stream_process(send_raw=params["send_raw"], send_processed=params["send_processed"], send_features=params["send_features"], overlap=overlap, filters=config.filters, features=config.features, eeg_channels=config.channels_eeg)


        self.board_handle.start_stream()
        self.board_streaming = True


    def stream_process(self, send_raw:bool, send_processed:bool, send_features:bool, overlap:bool, filters, features, eeg_channels):
        packet = self.stream_reader.get_data_packet()
        send_processed = True
        send_raw = False
        if packet.shape[0] != 0:

            if send_raw:
                self.queue_raw.put_nowait(packet)

            if (send_features or send_processed):
                # Returns a <feature_size> sized array of data.
                # If overlap is true it will return a full array with every new packet, 
                # If overlap is false it will only return once every <feature_size/packet_size> packets.
                feature_size_packet = self.stream_reader.get_feature_size_packet(overlap)
                filtered_data = np.array(feature_size_packet, order="C")

                start_time = time.perf_counter()

                if send_processed and feature_size_packet.shape[0] != 0:
                    for filter in filters:
                        filter_function = get_filter_function(filter)
                        for channel in eeg_channels:
                            filter_function(filtered_data[channel])
    
                    end_time = time.perf_counter()

                    print(f"Filtering time: {(end_time - start_time)*1000:.3f} ms")

                    try:
                        self.queue_raw.put_nowait(filtered_data if not overlap else filtered_data[-self.stream_reader.packet_size:])
                    except:
                        pass

                if send_features:

                    pass


    def stop_stream(self):
        if self.board_streaming:
            self.board_handle.stop_stream()
            self.board_handle.release_all_sessions()
            self.board_streaming = False

    def disconnect(self):
        self.stop_stream()
        self.board_connected = False
        self.board_handle = None

    def process_commands(self):
        cmd, data = self.get_cmd()

        if cmd == "connect":
            self.connect(data)
        if cmd == "start_stream":
            self.start_stream(data)
        if cmd == "stop_stream":
            self.stop_stream()
        if cmd == "disconnect":
            self.disconnect()

    def get_cmd(self):
        try:
            data = self.queue_cmd_rx.get_nowait()
        except Empty:
            return None, None

        if isinstance(data, list) or isinstance(data, tuple):
            return data
    
        return data, None
    
