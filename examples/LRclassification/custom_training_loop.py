import os
import shutil
import mybci
import pprint
import json

import numpy as np

from keras import models

# Paths to the models
model_paths = ["F1/ML1", "F2/ML1", "F3/ML1"]
model_paths = ["data/datasets/LRClassification/" + path for path in model_paths]

batch_size = 32
val_split = 0.2


# List of epoch sizes to test
epoch_sizes = [5, 10]  # Add your desired epoch sizes here

# Number of times to repeat training and testing for each combination
num_repeats = 5


# Directory names for train and test datasets
train_dir = "train/"
test_dir = "test/"

# List of substrings to exclude from training datasets
exclude_substrings = []

# Function to check if a file should be excluded based on substrings
def should_exclude(filename):
    for substring in exclude_substrings:
        if substring in filename:
            return True
    return False

# Function to copy datasets to train and test directories
def copy_datasets(train_files, test_file, src_path):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Clear existing files in the directories
    for f in os.listdir(train_dir):
        os.remove(os.path.join(train_dir, f))
    for f in os.listdir(test_dir):
        os.remove(os.path.join(test_dir, f))

    # Copy training files, excluding those with specific substrings
    for file in train_files:
        if not should_exclude(file):
            shutil.copy(os.path.join(src_path, file), train_dir)
    
    # Copy testing file
    shutil.copy(os.path.join(src_path, test_file), test_dir)

def testing(model, test_dict, actions):
    results = {}
    for dataset_file, test_data in test_dict.items():
        results[dataset_file] = {}
        for i in actions:
            x1 = np.array([data['fftdata'] for data in test_data[i]])
            x2 = np.array([data['timeseries'] for data in test_data[i]])
            y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data[i]])
            temp_results = model.evaluate([x1, x2], y, return_dict=True)
            results[dataset_file][f'{i}'] = temp_results

    return results

# Loop over each model directory
for model_path in model_paths:
    datasets = os.listdir(model_path)
    n = len(datasets)
    
    # Loop to exclude each dataset once
    for excluded_index in range(n):
        excluded_dataset = datasets[excluded_index]
        
        # Filter out the excluded dataset
        available_datasets = datasets[:excluded_index] + datasets[excluded_index + 1:]
        
        # Loop over each epoch size
        for epochs in epoch_sizes:
            # Dictionary to hold all results for the current epoch size and exclusion
            all_results = {}

            # Iterate over each possible combination
            for i in range(len(available_datasets)):
                # Select one dataset for testing
                test_file = available_datasets[i]
                
                # Select the rest for training
                train_files = available_datasets[:i] + available_datasets[i + 1:]
                
                # Copy files to respective directories
                copy_datasets(train_files, test_file, model_path)
                
                # List to hold results for this specific combination
                combination_results = []

                # Repeat training and testing num_repeats times
                for repeat in range(num_repeats):
                    # Call your function to load data and train the model here
                    train_data, _ = mybci.load_and_split_data(train_dir, 0.0, True)
                    _, test_dict = mybci.load_and_split_data(test_dir, 0.7, True)
                    model_index = model_paths.index(model_path) + 1
                    print(f"Training Model {model_index} with dataset combination where {test_file} is used for testing, {excluded_dataset} is excluded, with {epochs} epochs. Repeat {repeat + 1}/{num_repeats}")

                    model = models.load_model("models/base/32_128_model_2.keras")
                    X1 = np.array([data['fftdata'] for data in train_data])
                    X2 = np.array([data['timeseries'] for data in train_data])
                    Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in train_data])
                    model.fit([X1, X2], Y, batch_size=batch_size, epochs=epochs, validation_split=val_split)

                    results = testing(model, test_dict, [1, 2])

                    # Append the results of this repeat to the combination results
                    combination_results.append(results)

                # Store combination results in all_results dictionary
                all_results[f"Model_{model_index}_Test_{test_file}_Excluded_{excluded_dataset}"] = {
                    "train_files": train_files,
                    "test_file": test_file,
                    "results": combination_results,
                    "epochs": epochs,
                    "excluded": excluded_dataset
                }

            # Save all results for the current epoch size and exclusion to a file
            result_filename = f"results/results_epochs_{epochs}_excluded_{excluded_dataset}.json"
            with open(result_filename, "w") as f:
                json.dump(all_results, f, indent=4)

            print(f"Results saved to {result_filename}")

