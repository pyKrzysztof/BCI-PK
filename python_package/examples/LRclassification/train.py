import os
import sys
import mybci
import numpy as np
from keras import models, utils
import random as rd
import pprint
import json


# params
epochs = 11
batch_size = 32
passes = 10
save_acc_threshold = 0.75
models_path = "models/acc75/"

# utils.set_random_seed(4)
datasets_path = "data/datasets/LRClassification/"
datasets_directories = [f"F{i}/ML1/" for i in range(1, 5)]
my_models = {name: None for name in datasets_directories}
print(datasets_directories)


def testing(model, test_dict, actions, results_dict):
    for dataset_file, test_data in test_dict.items():
        results_dict[dataset_dir][dataset_file] = {}
        for i in actions:
            print(f"testing model {dataset_dir} on data from {dataset_file} action {i}:")
            x1 = np.array([data['fftdata'] for data in test_data[i]])
            x2 = np.array([data['timeseries'] for data in test_data[i]])
            y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data[i]])
            temp_results = model.evaluate([x1, x2], y, return_dict=True)
            results_dict[dataset_dir][dataset_file][f'{i}'] = temp_results
            if temp_results['accuracy'] < save_acc_threshold:
                print(f"Skipping {dataset_dir} based on low accuracy on {dataset_file} action {i}\n\n\n")
                return False

    return True

# multiple seeded sessions (seeds for equal dataset shuffles)
for index in range(1, passes+1):
    test_results = {name: {} for name in datasets_directories}
    # mybci.dataset_loading.seed = index
    mybci.dataset_loading.seed = index

    # training
    for dataset_dir in datasets_directories:
        train, test = mybci.load_and_split_data(os.path.join(datasets_path, dataset_dir), split=0.2, load_all=True)
        X1 = np.array([data['fftdata'] for data in train])
        X2 = np.array([data['timeseries'] for data in train])
        Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in train])
        model = models.load_model("models/base/32_128_model_2.keras")
        model.fit([X1, X2], Y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
        my_models[dataset_dir] = model

        # testing - returns True when model meets accuracy threshold on all tests.
        if testing(model, test, [1, 2], test_results):
            name = dataset_dir.replace('/', '_')
            name = f"{name}_{index}.keras"
            path = os.path.join(models_path, name)
            print(f"Saving model: {name}")
            models.save_model(model, path)

        print(f"Pass # {index}")
        pprint.pprint(test_results)



        with open(f"temp_results/results{index}.json", 'w') as f:
            json.dump(test_results, f, sort_keys=True, indent=4)

        


# models.save_model(model1, "models/F1_model.keras")