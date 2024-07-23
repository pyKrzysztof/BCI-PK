import os
import pickle
import random
import numpy as np

import enum

def shuffle_columns(data, seed=None):
    if seed is not None:
        rd = random.Random(seed)
    else:
        rd = random

    rows = list(zip(*data))
    rd.shuffle(rows)
    shuffled_data = list(zip(*rows))
    shuffled_data = [list(column) for column in shuffled_data]

    return shuffled_data

def load_single_dataset(filepath, extend:dict={}):
    """
    Loads training data from a single pickle file.

    Parameters:
        - filepath (str): Path to the pickle file.

    Returns:
        - A dictionary with group labels as keys and lists of data entries as values.
    """

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        for label, group_data in data.items():
            if label not in extend:
                extend[label] = []
            extend[label].extend(group_data)
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
            load_single_dataset(filepath, all_data)
            if verbose:
                print("Loaded:", filename)
    if load_excluded:
        return all_data, {name: load_single_dataset(os.path.join(directory_path, name)) for name in excluded_filenames}
    return all_data

def split_dataset_to_xy(data, xlabels:list, label_func:any, grouped=True, split=0.2, seed=None, excluded_groups=[]):
    if seed is not None:
        rd = random.Random(seed)
    else:
        rd = random

    xlabels = [label.name for label in xlabels if isinstance(label, enum.Enum)]

    train_dataset = []
    validation_dataset = []

    if grouped:
        for group_label, group_data in data.items():
            if group_label in excluded_groups:
                continue
            temp = group_data
            rd.shuffle(temp)
            split_index = int(len(group_data) * split)
            train_dataset.extend(group_data[split_index:])
            validation_dataset.extend(group_data[:split_index])
    else:
        split_index = int(len(data) * split)
        train_dataset.extend(data[split_index:])
        validation_dataset.extend(data[:split_index])

    rd.shuffle(train_dataset)
    rd.shuffle(validation_dataset)

    if not label_func:
        label_func = lambda _: _

    if len(xlabels) != 1:
        return ( [np.array([data[datatype] for data in train_dataset]) for datatype in xlabels], np.array([label_func(label) for label in [data['label'] for data in train_dataset]]) ), \
                ( [np.array([data[datatype] for data in validation_dataset]) for datatype in xlabels], np.array([label_func(label) for label in [data['label'] for data in validation_dataset]]) )

    return ( *[np.array([data[datatype] for data in train_dataset]) for datatype in xlabels], np.array([label_func(label) for label in [data['label'] for data in train_dataset]]) ), \
                    ( *[np.array([data[datatype] for data in validation_dataset]) for datatype in xlabels], np.array([label_func(label) for label in [data['label'] for data in validation_dataset]]) )


def dataset_ungrouped_to_xy(dataset, xlabels:list, label_func:any, shuffle=False, seed=None):
    if shuffle:
        if seed is not None:
            rd = random.Random(seed)
        else:
            rd = random
        rd.shuffle(dataset)
    return ( [np.array([data[datatype] for data in dataset]) for datatype in xlabels], np.array([label_func(label) for label in [data['label'] for data in dataset]]) )

def combine_xy_datasets(*datasets_xy, shuffle=True, seed=None):
    #[ (x, y), (x, y), ]     x = [ [x1...] , [x2...], ... ]
    X = []
    Y = []

    for xy in datasets_xy:
        for i in range(len(xy[0])):
            if len(X) <= i:
                X.append([])
            X[i].extend(xy[0][i])
        Y.extend(xy[1])

    assert len(X[0]) == len(Y)

    if shuffle:
        dataset = shuffle_columns([*X, Y], seed)
    
    Y_out = np.array(dataset[-1])
    X_out = list([np.array(data) for data in dataset[:-1]])
    return (X_out, Y_out)


def split_dataset(dataset, split, preshuffle=False, postshuffle=True, grouped=True, seed=None):
    if seed is not None:
        rd = random.Random(seed)
    else:
        rd = random

    dataset_1 = {}
    dataset_2 = {}
    if grouped:
        for group_label, group_data in dataset.items():
            temp = group_data
            if preshuffle:
                rd.shuffle(temp)
            split_index = int(len(group_data) * split)
            dataset_1[group_label] = temp[split_index:]
            dataset_2[group_label] = temp[:split_index]
            if postshuffle:
                rd.shuffle(dataset_1[group_label])
                rd.shuffle(dataset_2[group_label])
            assert group_data[split_index:] != dataset_1[group_label]
            assert group_data[:split_index] != dataset_2[group_label]
    else:
        split_index = int(len(dataset) * split)
        dataset_1.extend(dataset[split_index:])
        dataset_2.extend(dataset[:split_index])

    
    return dataset_1, dataset_2





def apply_func_to_dataset(dataset, func, feature_names):
    """ func will be passed data for provided feature_names and has to return the changed data in that order. """
    # not implemented yet
    pass
    


