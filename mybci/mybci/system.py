"""
Custom headset ideas:
https://www.thingiverse.com/thing:139725
https://conorrussomanno.wordpress.com/2013/12/17/3d-printed-eeg-headset-v1-aka-spiderclaw/
"""
import os
import sys
# import dill
import stat
import pickle
import pathlib
import importlib
import numpy as np
import pandas as pd

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List

from brainflow import BoardIds, BoardShim

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

    filters: list[Filters] = field(default_factory = lambda: [])
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
    base_path = None
    models_path = None
    data_path = None
    temp_path = None
    system_file_path = None
    action_data_path = None
    layouts_path = None
    pipelines_path = None

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
    
    def __init__(self, data_source: any, packet_size: int, feature_size: int, sep='\t'):
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

    config: SystemConfig = None
    stats: Statistics = None
    files: Filesystem = None

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
        system.create_filesystem()
        system.save_system()

        with open("run.sh", 'w') as f:
            f.writelines("python -i -c \"from mybci.system import System;sys = System.load_system();print('System loaded into variable <sys>.')\"")
        st = os.stat("run.sh")
        os.chmod("run.sh", st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        

    def load_system(self, path:str = None):
        """Loads saved system."""
        if not path:
            path = os.getcwd() + "/"

        try:
            from config import config # type: ignore
        except:
            raise
        finally:
            self.config = config

        with open(path + "system.bin", 'rb') as f:
            self.stats, self.files = pickle.load(f)

        self.create_filesystem() # in case config changed
        return self


    def save_system(self):
        """Saves system to a binary file. """
        with open (self.files.system_file_path, 'wb') as f:
            pickle.dump([self.stats, self.files], f)

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
