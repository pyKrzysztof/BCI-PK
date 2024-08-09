import time
import math
import datetime
import collections
import numpy as np
import dearpygui.dearpygui as dpg

from queue import Queue, Empty
from multiprocessing.synchronize import Event
from brainflow import BoardIds, BoardShim, BrainFlowInputParams

from mybci.system import System, SystemConfig, Statistics, Filesystem, DataReader, DataProcessor
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

        with dpg.window(label='EEG', tag='eegwin', pos=(1100, 20), width=480, height=-1, collapsed=True, no_close=True):
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

    def set_reinforcement_handler(self, func):
        self._reinforcement_callback_func = func

    def get_reinforcement_callback(self):
        return self._reinforcement_callback_func

    def create_training_window(self):
        indent = 10
        # with dpg.window(label="Training", tag="training-window", width=400, height=600, no_close=True, pos=(20, 400)):
        with dpg.collapsing_header(label="Recording & Training", tag="training-header", indent=0):

            # Pipeline Header
            with dpg.collapsing_header(label="Pipeline", parent="training-header", tag="pipeline-header", indent=indent):

                dpg.add_input_int(label="Repetitions per action", tag="reps", default_value=10, width=100, indent=indent)
                dpg.add_input_text(label="Actions to classify", tag="actions-train", default_value=", ".join(self.system.config.actions), width=200, indent=indent)
                dpg.add_input_int(label="Min. action duration [s]", tag="action-t0", default_value=4, width=100, indent=indent)
                dpg.add_input_int(label="Max. action duration [s]", tag="action-t1", default_value=6, width=100, indent=indent)

                # Advanced Header
                with dpg.collapsing_header(label="Advanced", parent="pipeline-header", indent=indent):

                    dpg.add_checkbox(label="App supports reinforcement.", default_value=True,
                                    callback= lambda _, state: setattr(self, "use_reinforcement", state), indent=2*indent)

                    with dpg.file_dialog(directory_selector=False, show=False, callback=file_open_callback,  tag="custom_callback_script_dialog", width=700 ,height=400):
                        dpg.add_file_extension(".py", color=(20, 80, 180, 255), custom_text="[PYTHON]")

                    dpg.add_button(label="Custom action callback script", width=-1, callback=lambda: dpg.show_item("custom_callback_script_dialog"), tag="custom_callbacks_select_button", enabled=True, indent=2*indent)


                def start_recording():
                    # custom_callbacks_script = dpg.get_value()
                    custom_callbacks = False # to be implemented, currently redundant with reinforcement.

                    pipeline_settings = {
                        "config": self.system.config,
                        "n": dpg.get_value("reps"),
                        "actions": dpg.get_value("actions-train"),
                        "t_min": dpg.get_value("action-t0"),
                        "t_max": dpg.get_value("action-t1"),
                        "send_reincorement_callbacks": self.use_reinforcement,
                        "on_init": None if not custom_callbacks else "",
                        "on_exit": None if not custom_callbacks else "",
                        "on_action": None if not custom_callbacks else "",
                        "finish_callback": True,
                        "connect": {
                            "board_port": self.system.config.board_port,
                            "board_id": dpg.get_value("board-id-selector")
                        },
                    }

                    dpg.disable_item("record-button")
                    dpg.set_item_label("record-button", "Recording..")

                    self.send_cmd(("pipeline", pipeline_settings))


                dpg.add_button(label="Record", tag="record-button", callback=start_recording)

            # Models Header
            with dpg.collapsing_header(label="Models", parent="training-header", tag="models-header", indent=indent):

                model_names = [model for model in self.system.model_handler.base_models.keys()]

                def model_base_selector_callback(selected_model):
                    dpg.set_value("base-model-description", self.system.model_handler.base_models[selected_model]["description"])
                    dpg.configure_item("model-selector", items=[str(index) for index in self.system.model_handler.model_dict[selected_model].keys()])

                dpg.add_listbox(model_names, label="Base models", tag="base-model-selector", callback= lambda a, s: model_base_selector_callback(s))
                dpg.add_text(label="Model description placeholder.", tag="base-model-description")

                def model_selector_callback(index):
                    base = dpg.get_value("base-model-selector")
                    index = int(index)

                    print(base, index, self.system.model_handler.model_dict[base][index])



                dpg.add_text(label="Existing models for selected base:")
                dpg.add_listbox(tag="model-selector", callback= lambda a, s: model_selector_callback(s), width=-1)

                def create_new_model_from_base():
                    model_base = dpg.get_value("base-model-selector")
                    self.system.model_handler.create_model_from_base(model_base)
                    self.system.save_system()
                    dpg.configure_item("model-selector", items=[str(index) for index in self.system.model_handler.model_dict[model_base].keys()])

                dpg.add_button(label="New model from base", width=-1, tag="new-model-button", callback=create_new_model_from_base)



    def create_system_windows(self):
        """Context must be created before calling this function."""

        with dpg.window(label="System Panel", no_close=True, pos=(20, 20), width=400, height=700, no_move=True, no_resize=True, collapsed=False):

            # Save system header
            with dpg.collapsing_header(label="Save System"):

                dpg.add_button(label="Save System", callback=self.system.save_system, width=-1)

            # System information header
            with dpg.collapsing_header(label="System Information"):

                dpg.add_text(f"Actions: {self.system.config.actions}", tag="info_actions")

            # EEG Board Header
            with dpg.collapsing_header(label="EEG Board", default_open=True):

                dpg.add_input_text(label="Board Port", default_value=self.system.config.board_port, tag="board_port",
                                    callback=lambda _, value: setattr(self.system.config, "board_port", value))

                dpg.add_listbox([self.system.config.board_type, BoardIds.SYNTHETIC_BOARD], tag="board-id-selector") # type: ignore

                dpg.add_button(label="Connect", width=-1, tag="board-connect-button",
                                callback=lambda: self.send_cmd(["connect", {"board_port": self.system.config.board_port, "board_id": dpg.get_value("board-id-selector")}]))

                dpg.add_button(label="Start Stream", width=-1, tag="board-start-button",
                                callback=lambda: self.send_cmd(["start_stream", {"session": "standard", "send_raw": True, "send_processed": True, "send_features": True, "config": self.system.config}]))

                dpg.add_button(label="Stop Stream", width=-1, tag="board-stop-button",
                                callback=lambda: self.send_cmd("stop_stream"))

                dpg.add_button(label="Disconnect", width=-1, tag="board-disconnect-button",
                                callback=lambda: self.send_cmd("disconnect"))

            # Training Windows
            self.create_training_window()

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

            if queue_updates["cmd"] is not None:
                cmd = queue_updates["cmd"]
                print(cmd)
                if cmd[0] == "reinforcement":
                    status = cmd[1]

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
