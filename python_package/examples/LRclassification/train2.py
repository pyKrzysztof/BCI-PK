import os
import sys
import mybci
import numpy as np
from keras import models, utils
import random as rd
import pprint
import json


# params
epochs = 7
batch_size = 32
passes = 5
save_acc_threshold = 0.0

model_paths = [f"models/base/model_{i}_4.keras" for i in [2]]

# utils.set_random_seed(4)
datasets_path = "data/datasets/LRClassification2/"
datasets_directories = [f"F{i}/ML1/" for i in [1,]]
my_models = {name: None for name in datasets_directories}
print(datasets_directories)

def testing(model, test_dict, actions, results_dict):
    for dataset_file, test_data in test_dict.items():
        for i in actions:
            # print(f"testing model {model} on data from {dataset_file} action {i}:")
            x1 = np.array([data['fftdata'] for data in test_data[i]])
            x2 = np.array([data['timeseries'] for data in test_data[i]])
            y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data[i]])
            temp_results = model.evaluate([x1, x2], y, return_dict=True)
            results_dict[f'{i}'] = temp_results
            if temp_results['accuracy'] < save_acc_threshold:
                print(f"Skipping {dataset_dir} based on low accuracy on {dataset_file} action {i}\n\n\n")
                return False
    return True

def train_model(model_path, train_data):
    X1 = np.array([data['fftdata'] for data in train_data])
    X2 = np.array([data['timeseries'] for data in train_data])
    Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in train_data])
    model = models.load_model(model_path)
    model.fit([X1, X2], Y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    return model

# multiple seeded sessions (seeds for equal dataset shuffles)
for index in range(1, passes+1):
    test_results = {model_path: {} for model_path in model_paths}
    # test_results['excluded']
    # mybci.dataset_loading.seed = index
    mybci.dataset_loading.seed = index

    # training
    for dataset_dir in datasets_directories:
        for model_path in model_paths:
            train, _ = mybci.load_and_split_data(os.path.join(datasets_path, dataset_dir), split=0.0, load_all=True)
            model = train_model(model_path, train)
            _, test = mybci.load_and_split_data(os.path.join(datasets_path, dataset_dir, "test/"), split=1.0, load_all=True)
            # testing - returns True when model meets accuracy threshold on all tests.
            if testing(model, test, [1, 2], test_results[model_path]):
                pass

            print(f"Pass # {index}")
            pprint.pprint(test_results)

        path = f"temp_results/models234_{epochs}epochs_teston_2006_5/"
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'results{index}.json'), 'w') as f:
            json.dump(test_results, f, sort_keys=True, indent=2)
