

BASE_CONFIG = {

    'name': '', # name of the config / data, will be used to save the final datasets.

    # live board data config - not implemented yet.
    'board_device': '/dev/ttyUSB0', # board connection physical port.
    'use_board_device': False, # whether to use the board for processing (toggles live data processing / file data processing).
    'electrode_config': [],
    'save_raw_session': True, # when in live data processing, toggles raw session data file saving.
    'live_pipeline': None, # when in live data processing mode, specifies (optionally) a pipeline for training, blocks live prediction.
    'prediction_functions': [], # functions that take current training packet data (output of ml_prepare_func)
    
    # File playback config
    'session_file': '',

    # common data processing config
    'action_markers': [], # markers of the data to be processed.
    'buffer_size': 128, # data buffer size, for default behaviour should be the biggest of the three (packet_size, filter_size, ml_prepare_size)
    'channel_column_ids': [1, 2, 3, 4, 5, 6, 7, 8], # dataframe / array columns to be passed to the filtering function.
    'packet_size': 32, # how many data samples are to be received at a time.
    'filter_size': 128, # amount of packets to be passed to the filter function.
    'filter_func': {}, # can return any amount of data rows but only last 'packet_size' entries will be saved. Multiple filtering functions can be passed to generate more data.
    'filter_func_extra_columns': [], # extra columns to pass to the filtering function, eg. accel channels.
    'ml_prepare_size': 128, # amount of samples to be sent to the ml_prepare_func
    'ml_prepare_chunk_start_offset': 250, # amount of samples to pass after chunk starts before starting the 'ml_prepare_func' (eg. if the chunk starts at packet no. 0, the first call to ml_prepare_func will be with packets no. 0 + 'ml_prepare_chunk_start_offset' - 'ml_prepare_size'. Main use is consideration for human reaction time in training.
    'ml_prepare_func': {}, # a dictionary of functions that have to return a dictionary of ''file type': data' key-value pairs for passed data, the output data will be saved and passed as ML data.
    'ml_prepare_extra_columns': [],
    # example usage would be generating fftdata from timeseries data and saving both data to be used for training.
    # multiple functions can be passed for creating many datasets with different data.

    # custom data processing config - only makes sense for pipelined / labeled data.
    'chunk_func_pass_extra_columns': [], # extra columns to pass to the chunk processing function, eg. accel channels.
    'chunk_func': None, # a function that gets passed COMPLETE single chunk data (from live pipeline processing or offline processing) to be used when writing custom behaviour. 

    # filesystem config
    'output_path_chunks': 'data/',
    'output_path_training_data': 'data/',
    'output_path_training_dataset': 'data/datasets/',
    'save_chunks': False, # saves chunks to a folder, could be useful when making many changes to processing functions on huge amount of data.
    'keep_seperate_training_data': False, # if True, it will not remove the intermediate files for training dataset making.
    'save_training_dataset': True, # it's possible to disable it but why would you use this processor then.
    'sep': '\t', # value separator in data files.

}


def get_base_config() -> dict:
    """
## Configuration Parameters

### General Config
- `name`: ''  \n
  Name of the config/data, will be used to save the final datasets.

### Live Board Data Config (not implemented yet)
- `board_device`: '/dev/ttyUSB0'  \n
  Board connection physical port.
- `use_board_device`: False  \n
  Whether to use the board for processing (toggles live data processing/file data processing).
- `save_raw_session`: True  \n
  When in live data processing, toggles raw session data file saving.
- `live_pipeline`: None  \n
  When in live data processing mode, specifies (optionally) a pipeline for training, blocks live prediction.
- `prediction_functions`: []  \n
  Functions that take current training packet data (output of ml_prepare_func).

### File Playback Config
- `session_file`: ''  

### Common Data Processing Config
- `action_markers`: []  \n
  Markers of the data to be processed.
- `buffer_size`: 128  \n
  Data buffer size, for default behavior should be the biggest of the three (packet_size, filter_size, ml_prepare_size).
- `channel_column_ids`: [1, 2, 3, 4, 5, 6, 7, 8]  \n
  DataFrame/array columns to be passed to the filtering function.
- `packet_size`: 32  \n
  Number of data samples to be received at a time.
- `filter_size`: 128  \n
  Amount of packets to be passed to the filter function.
- `filter_func`: {}  \n
  Can return any amount of data rows but only last `packet_size` entries will be saved. Multiple filtering functions can be passed to generate more data.
- `filter_func_extra_columns`: []  \n
  Extra columns to pass to the filtering function, e.g., accel channels.
- `ml_prepare_size`: 128  \n
  Amount of samples to be sent to the ml_prepare_func.
- `ml_prepare_chunk_start_offset`: 250  \n
  Amount of samples to pass after chunk starts before starting the ml_prepare_func (e.g., if the chunk starts at packet no. 0, the first call to ml_prepare_func will be with packets no. 0 + ml_prepare_chunk_start_offset - ml_prepare_size). Main use is consideration for human reaction time in training.
- `ml_prepare_func`: {}  \n
  A dictionary of functions that have to return a dictionary of `'file type': data` key-value pairs for passed data. The output data will be saved and passed as ML data.
- `ml_prepare_extra_columns`: []  \n
  Example usage would be generating FFT data from timeseries data and saving both data to be used for training. Multiple functions can be passed for creating many datasets with different data.

### Custom Data Processing Config (only makes sense for pipelined/labeled data)
- `chunk_func_pass_extra_columns`: []  \n
  Extra columns to pass to the chunk processing function, e.g., accel channels.
- `chunk_func`: None  \n
  A function that gets passed COMPLETE single chunk data (from live pipeline processing or offline processing) to be used when writing custom behavior.

### Filesystem Config
- `output_path_chunks`: 'data/chunks/'  
- `output_path_training_data`: 'data/training/'  
- `output_path_training_dataset`: 'data/datasets/'  
- `save_chunks`: False  \n
  Saves chunks to a folder, could be useful when making many changes to processing functions on huge amounts of data.
- `keep_separate_training_data`: False  \n
  If True, it will not remove the intermediate files for training dataset making.
- `save_training_dataset`: True  \n
  It's possible to disable it but why would you use this processor then.
- `sep`: '\t'  \n
  Value separator in data files.

    """
    return dict(BASE_CONFIG)
