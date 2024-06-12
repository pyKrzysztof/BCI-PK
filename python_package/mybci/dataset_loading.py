import os
import pickle
import random



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



def load_all_datasets(directory_path):
    """
    Loads training data from all pickle files in a directory.

    Parameters:
        - directory_path (str): Path to the directory containing pickle files.

    Returns:
        - A dictionary with group labels as keys and lists of data entries as values.
    """
    all_data = {}

    # Load all pickle files in the specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                for label, group_data in data.items():
                    if label not in all_data:
                        all_data[label] = []
                    all_data[label].extend(group_data)

    return all_data

def shuffle_and_split_group(group_data, split):
    """
    Shuffles and splits a single group's data into training and test sets.

    Parameters:
        - group_data (list): List of data entries for the group.
        - split (float): Split factor from 0.0 to 1.0, the number represents the % of data to be test data.

    Returns:
        - A tuple (train_data, test_data) for the group.
    """
    random.shuffle(group_data)
    split_index = int(len(group_data) * split)
    return group_data[split_index:], group_data[:split_index]

def load_and_split_data(path, split, load_all=True):
    """
    Loads training data from .pickle files in a directory or a single file, shuffles the data, and splits into train and test data.

    Parameters:
        - path (str): Path to the directory or single file.
        - split (float): Split factor from 0.0 to 1.0, the number represents the % of data to be test data.
        - load_all (bool): If True, load all datasets from the directory; if False, load a single dataset file.

    Returns:
        - A tuple (train_data, test_data) where train_data is a list of dictionaries and test_data is a dictionary with group labels as keys and lists of data entries as values.
    """
    if load_all:
        data = load_all_datasets(path)
    else:
        data = load_single_dataset(path)

    train_data = []
    test_data = {label: [] for label in data.keys()}

    for group_label, group_data in data.items():
        group_train_data, group_test_data = shuffle_and_split_group(group_data, split)
        train_data.extend(group_train_data)
        test_data[group_label].extend(group_test_data)

    return train_data, test_data
