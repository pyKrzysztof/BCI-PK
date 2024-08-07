"""
Custom headset ideas:
https://www.thingiverse.com/thing:139725
https://conorrussomanno.wordpress.com/2013/12/17/3d-printed-eeg-headset-v1-aka-spiderclaw/
"""
import os
import sys
# import dill
import time
import stat
import pickle
import pathlib
import datetime
import importlib
from keras.src.models import Model
import numpy as np
import pandas as pd

import tensorflow as tf
from keras import models
from collections import deque
from queue import Queue, Empty
from typing import Callable, List
from dataclasses import dataclass, field
from multiprocessing.synchronize import Event
from brainflow import BoardIds, BoardShim, BrainFlowInputParams
from mybci.feature_extraction import Filters, Features, get_filter_function



@dataclass
class SystemConfig:

    path: str = ''
    name: str = 'Data Processing'
    board_type = BoardIds.CYTON_BOARD
    board_port: str = '/dev/ttyUSB0'

    actions: list[str] = field(default_factory = lambda: [])
    packet_size: int = 32
    feature_size: int = 128
    sampling_rate: int = 250
    process_overlaps: bool = False

    channels_accel: list[int] = field(default_factory = lambda: [10, 11, 12]) # chyba?
    channels_eeg: list[int] = field(default_factory = lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    electrode_config: list = field(default_factory = lambda: [])

    filters: list = field(default_factory = lambda: [])
    features: list[Features] = field(default_factory = lambda: [])

    separator: str = '\t'

    def create_config_file(self):
        data = f"""from mybci.system import SystemConfig
from mybci.feature_extraction import Filters, Features
from brainflow import BoardIds

config = SystemConfig()
config.path = "{self.path}"
config.name = "{self.name}"
config.board_type = {self.board_type.name}
config.board_port = "{self.board_port}"

config.actions = {[f"{action}" for action in self.actions]}
config.packet_size = {self.packet_size}
config.feature_size = {self.feature_size}
config.sampling_rate = {self.sampling_rate}
config.process_overlaps = {self.process_overlaps}

config.channels_accel = {self.channels_accel}
config.channels_eeg = {self.channels_eeg}
config.electrode_config = {self.electrode_config}

config.filters = {[filter.name for filter in self.filters]}
config.features = {[feature.name for feature in self.features]}

config.separator = "{self.separator}"
"""
        with open(self.path + "config.py", 'w') as f:
            f.write(data)


class Statistics:
    action_count = None
    data_on_day = None

    def __init__(self) -> None:
        self.action_count = {}
        self.data_on_day = {}


class Filesystem:
    base_path = ""
    models_path = ""
    data_path = ""
    temp_path = ""
    system_file_path = ""
    action_data_path = ""
    layouts_path = ""
    pipelines_path = ""

    def __init__(self) -> None:
        self.base_path = ""
        self.models_path = ""
        self.data_path = ""
        self.temp_path = ""
        self.system_file_path = ""
        self.layouts_path = ""
        self.pipelines_path = ""
        self.action_data_path = {}


class DataReader:

    def __init__(self, data_source, packet_size: int, feature_size: int, sep='\t'):
        self.data_source = data_source
        self.packet_size = packet_size
        self.feature_size = feature_size
        self.buffer = deque(maxlen=feature_size)
        self.sep = sep

        self.packets_to_filled = feature_size // packet_size

        if isinstance(self.data_source, BoardShim):
            self.board = self.data_source
            self.get_data_packet = self._get_live_data
            self.name = "live"
        elif isinstance(self.data_source, str):
            self.filepath = self.data_source
            self.data_iter = self._file_data_generator()
            self.get_data_packet = self._get_file_data
            self.name = os.path.splitext(os.path.basename(data_source))[0]

    def get_feature_size_packet(self, return_overlaps):
        if not return_overlaps:
            if self.packets_to_filled == 0:
                self.packets_to_filled = self.feature_size // self.packet_size
                return np.array(self.buffer)
        else:
            if len(self.buffer) >= self.feature_size:
                return np.array(self.buffer)

        return np.array([])

    def _get_live_data(self):
        if self.board.get_board_data_count() >= self.packet_size:
            if self.packets_to_filled == 0:
                self.packets_to_filled = self.feature_size // self.packet_size
            data = self.board.get_board_data(self.packet_size).transpose()
            self.buffer.extend(data)
            self.packets_to_filled -= 1
            return data
        else:
            return np.array([]) # Not enough data available

    def _get_file_data(self):
        try:
            data = next(self.data_iter)
            self.buffer.extend(data)
            return data
        except StopIteration:
            return np.array([])  # No more data in the file

    def _file_data_generator(self):
        for packet in pd.read_csv(self.filepath, chunksize=self.packet_size, sep=self.sep):
            yield packet.values



class ModelHandler:

    base_models: dict[str, dict] = {} # name: {description: str, features: list[Features], func: Callable}
    model_dict: dict[str, dict[int, dict]] = {} # name: {index: {path: path, stats: stats}}


    def __init__(self) -> None:
        pass

    def load_base_models(self, models_path):
        self.base_path = models_path
        for file in os.listdir(models_path):
            if not file.endswith(".py"):
                continue
            name = file.strip(".py")
            module = importlib.import_module(f"models.{name}")

            func:Callable = getattr(module, "get_model")
            features:list[Features] = getattr(module, "features")
            description:str = getattr(module, "description")

            self.base_models[name] = {
                "description": description,
                "features": features,
                "func": func
            }
            if name not in self.model_dict:
                self.model_dict[name] = {}

    def create_model_from_base(self, basename):
        if basename not in self.base_models:
            return

        index = 0
        while True:
            if index in self.model_dict[basename]:
                index += 1
                continue
            break

        path = os.path.join(self.base_path, basename + f"_{index}.keras")
        creator_func = self.base_models[basename]["func"]

        model: Model = creator_func()
        model.save(path)

        self.model_dict[basename][index] = {
            "path": path,
            "stats": {
                "times_trained": 0,
                "past_accuracy": [],
                "trained_on": [],
                "calibration_date": "",
            },
        }


    def train_model(self, basename, index, files):
        """Trains an existing indexed model of type <basename> using provided files."""
        model_info = self.base_models[basename]

        target_features = model_info["features"]

        if index != -1 and index not in self.model_dict[basename]:
            return "Model doesn't exist."

        # TODO: Finish this









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
        self.board_connected = False
        self.board_streaming = False
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
            self.board_handle.release_all_sessions()
            self.board_handle.prepare_session()
        except:
            try:
                self.queue_cmd_tx.put_nowait(["Error", "Failed to connect to the board."])
            except:
                pass

            self.board_connected = False

    def start_stream(self, params):
        if not self.board_connected:
            return

        self.board_handle.release_all_sessions()
        self.board_handle.prepare_session()

        # if not self.board_handle.is_prepared():
        #     self.board_handle.release_all_sessions()
        #     self.board_handle.prepare_session()

        name = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
        self.board_handle.add_streamer(f"file://data/temp/raw_{name}.csv:w")

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

        if cmd == "pipeline":
            print("Pipeline recording request:", data)

    def get_cmd(self):
        try:
            data = self.queue_cmd_rx.get_nowait()
        except Empty:
            return None, None

        if isinstance(data, list) or isinstance(data, tuple):
            return data

        return data, None

    def bind_model_predict(self):
        pass

    def bind_model_train(self):
        pass



class System:

    """
    filesystem tree:

    <base_path>/
        models/ - models go here, both clean and trained with their code / maps.

        data/

            <actions[n]>/
                raw/ - raw data of a single chunk of action data, before saving check quality of the signal (automatically)
                processed/ - data processed with current filters, including feature extraction.
            ...

            temp/ - temporary files go here

        <system.bin>
        <run.sh>
    """

    config: SystemConfig = SystemConfig()
    stats: Statistics = Statistics()
    files: Filesystem = Filesystem()
    model_handler: ModelHandler = ModelHandler()

    def __init__(self) -> None:
        pass

    @classmethod
    def create_system(cls):
        path = os.getcwd() + "/"
        print("Created system at current working directory:", path)
        system = System()
        system.config = SystemConfig(path)
        system.config.create_config_file()
        system.stats = Statistics()
        system.files = Filesystem()
        system.model_handler = ModelHandler()

        system.create_filesystem()
        system.model_handler.load_base_models(system.files.models_path)
        system.save_system()

        with open("run.sh", 'w') as f:
            f.writelines("python -i -c \"from mybci.system import System;sys = System.load_system();print('System loaded into variable <sys>.')\"")
        st = os.stat("run.sh")
        os.chmod("run.sh", st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


    def load_system(self, path:str = ""):
        """Loads saved system."""
        if not path:
            path = os.getcwd() + "/" # THIS MAKES THE SYSTEM OPENABLE ONLY FROM THE SYSTEM DIRECTORY.

        try:
            from config import config # type: ignore
        except:
            raise
        finally:
            self.config = config # type: ignore

        with open(path + "system.bin", 'rb') as f:
            self.stats, self.files, self.model_handler.model_dict = pickle.load(f)

        self.create_filesystem() # in case config changed / user deleted folders, files.
        self.model_handler.load_base_models(self.files.models_path)
        # self.model_handler.create_model_from_base("model_1")
        return self


    def save_system(self):
        """Saves system to a binary file. """
        with open (self.files.system_file_path, 'wb') as f:
            pickle.dump([self.stats, self.files, self.model_handler.model_dict], f)

    def create_filesystem(self):
        self.files.base_path = self.config.path
        self.files.models_path = os.path.join(self.files.base_path, "models/")
        self.files.data_path = os.path.join(self.files.base_path, "data/")
        self.files.temp_path = os.path.join(self.files.data_path, "temp/")
        self.files.system_file_path = os.path.join(self.files.base_path, "system.bin")
        self.files.layouts_path = os.path.join(self.files.base_path, "layouts/")
        self.files.pipelines_path = os.path.join(self.files.base_path, "pipelines/")
        os.makedirs(self.files.models_path, exist_ok=True)
        os.makedirs(self.files.data_path, exist_ok=True)
        os.makedirs(self.files.temp_path, exist_ok=True)
        os.makedirs(self.files.layouts_path, exist_ok=True)
        os.makedirs(self.files.pipelines_path, exist_ok=True)
        for action in self.config.actions:
            self.files.action_data_path[action] = os.path.join(self.files.data_path, action + "/")
            os.makedirs(os.path.join(self.files.action_data_path[action], "raw/"), exist_ok=True)
            os.makedirs(os.path.join(self.files.action_data_path[action], "processed/"), exist_ok=True)
