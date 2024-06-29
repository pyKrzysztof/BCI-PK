import mybci
from preprocess_functions import filter1, filter2, filter3, filter4, prep1, prep2 # type: ignore (vs code won't resolve this import)

config = mybci.get_base_config()

config['name'] = "processed_data"
config['action_markers'] = [1, 2]

config['session_file'] = []
config['session_file'].extend([f'data/session_data/0406_{i}.csv' for i in [1, 2]])
config['session_file'].extend([f'data/session_data/1806_{i}.csv' for i in range(1, 6)])
config['session_file'].extend([f'data/session_data/2006_{i}.csv' for i in range(1, 6)])

config['packet_size'] = 32

config['filter_size'] = 64
config['filter_func'] = {f'f{i+1}': filter for i, filter in enumerate([filter1, filter2, filter3, filter4])}

config['ml_prepare_size'] = 128
config['ml_prepare_func'] = {'p1': prep1, 'p2': prep2}

config['save_training_dataset'] = True
config['keep_seperate_training_data'] = False
config['output_path_training_dataset'] = "data/datasets/"

processor = mybci.DataProcessor(config)
processor.process()
