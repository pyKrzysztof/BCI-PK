import os
import mybci
import numpy as np
import pprint
from keras import models

model_src = "models/acc70/"

datasets_path = "data/test_datasets/LRClassification/"
datasets_directories = [f"F{i}/ML1/" for i in [1, 2]]
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

    return True

# multiple seeded sessions (seeds for equal dataset shuffles)
test_results = {name: {} for name in datasets_directories}

my_models = {}
results_dict = {name: {} for name in datasets_directories}
for model_name in os.listdir(model_src):
    if not model_name.endswith(".keras"):
        continue
    # print(model_name)
    key = model_name[:7].replace("_", "/")

    model = models.load_model(os.path.join(model_src, model_name))

    for dataset_dir in datasets_directories:
        path = os.path.join(datasets_path, dataset_dir)

        _, test_dict = mybci.load_and_split_data(path, 1.0, True)

        if key == dataset_dir:
            testing(model, test_dict=test_dict, actions=[1, 2], results_dict=results_dict)
            pprint.pprint(results_dict)