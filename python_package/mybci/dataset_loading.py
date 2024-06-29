import os
import pickle
import random
import numpy as np



def load_single_dataset(filepath):
    """
    Loads training data from a single pickle file.

    Parameters:
        - filepath (str): Path to the pickle file.

    Returns:
        - A dictionary with group labels as keys and lists of data entries as values.
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data



def load_all_datasets(directory_path, exclude_matching=None, load_excluded=False, verbose=False):
    """
    Loads training data from all pickle files in a directory.

    Parameters:
        - directory_path (str): Path to the directory containing pickle files.

    Returns:
        - A dictionary with group labels as keys and lists of data entries as values.
        - if 'load_excluded' is set to True it will also return a tuple containing excluded datasets.
    """
    all_data = {}
    excluded_filenames = []
    # Load all pickle files in the specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.pickle'):
            if exclude_matching is not None:
                if isinstance(exclude_matching, list) or isinstance(exclude_matching, tuple):
                    if sum([match in filename for match in exclude_matching]) > 0:
                        excluded_filenames.append(filename)
                        print("Excluded:", filename)
                        continue
                elif exclude_matching in filename:
                    excluded_filenames.append(filename)
                    if verbose:
                        print("Excluded:", filename)
                    continue

            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                for label, group_data in data.items():
                    if label not in all_data:
                        all_data[label] = []
                    all_data[label].extend(group_data)
            if verbose:
                print("Loaded:", filename)
    
    if load_excluded:
        return all_data, {name: load_single_dataset(os.path.join(directory_path, name)) for name in excluded_filenames}
    return all_data



def create_training_and_validation_datasets(data, xlabels:list, label_func:any, split=0.0, seed=None):
    if seed is not None:
        rd = random.Random(seed)
    else:
        rd = random

    train_dataset = []
    validation_dataset = []

    for group_label, group_data in data.items():
        temp = group_data
        rd.shuffle(temp)
        split_index = int(len(group_data) * split)
        train_dataset.extend(group_data[split_index:])
        validation_dataset.extend(group_data[:split_index])

    rd.shuffle(train_dataset)
    rd.shuffle(validation_dataset)

    return ( [np.array([data[datatype] for data in train_dataset]) for datatype in xlabels], np.array([label_func(label) for label in [data['label'] for data in train_dataset]]) ), \
            ( [np.array([data[datatype] for data in validation_dataset]) for datatype in xlabels], np.array([label_func(label) for label in [data['label'] for data in validation_dataset]]) )


def dataset_to_x_y(dataset, xlabels:list, label_func:any):
    return ( [np.array([data[datatype] for data in dataset]) for datatype in xlabels], np.array([label_func(label) for label in [data['label'] for data in dataset]]) )