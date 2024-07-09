import os
import pickle
import random
import mybci.training
import tensorflow as tf
import numpy as np
from keras import models, layers, utils
import json

import mybci

# seed = 2, unbiased, train_valid_split = 0.01
# best results so far for calib_split = 0.2: epochs 5 / 3,  
# best results so far for calib_split = 0.2: epochs 4 / 5, 
# calib_split = 0.15: epochs 2 / 3

# seed = 2, biased LR 1.0 / 1.10,  train_valid_split = 0.2
# calib_split = 0.15 (30 seconds): epochs 3 / 3

# those are best for now, those parameters do need fine tunning for new data, so need an algorithm to automatically set them later:
# epochs = 1
# epochs_calib = 3
# batch_size = 64
# batch_size_calib = 16
# train_valid_split = 0.2
# calib_split = 0.15
# bias_left = 1.0
# bias_right = 1.0

# params
epochs = 4
epochs_calib = 5
batch_size = 64
batch_size_calib = 16
train_valid_split = 0.2
calib_split = 0.15
bias_left = 1.0
bias_right = 1.0

def my_evaluate(X, Y, model):
    count_pos = 0
    count_neg = 0
    add_count_1 = 0
    add_count_2 = 0
    for indices, ys, results in mybci.training.custom_evaluate(X, Y, model, batch_size=4, step=4, return_indexed=True):
        sum_result = 0
        additive_result = np.array([0, 0], dtype=np.float64)

        for i, y, result in zip(indices, ys, results):
            additive_result += result*[bias_left, bias_right]
            sum_result += sum(result*[bias_left, -bias_right])
            if sum_result > 0:
                count_pos += 1
            else:
                count_neg += 1

        additive_result_y = additive_result.tolist().index(max(additive_result))
        if not additive_result_y:
            add_count_1 += 1
        else:
            add_count_2 += 1

        additive_result_y = [1 if not additive_result_y else 0, additive_result_y]
    
    result_1 = count_pos/(count_pos+count_neg)*100
    if Y[0].tolist() == [0, 1]:
        result_1 = 100 - result_1
    # print(f"Sum accuracy: {result_1:.2f}%")

    result_2 = add_count_1/(add_count_1+add_count_2)*100
    if Y[0].tolist() == [0, 1]:
        result_2 = 100 - result_2
    # print(f"Additive accuracy: {result_2:.2f}%")

    return result_1, result_2


# dataset directories by filter/process functions.
datasets_path = "data/datasets/processed_data/"
datasets_directories = [f"f{i}_p1" for i in [1]]
exclude_names = ["2004", "0406", "2006_5", "calib", "test"]

# functions
label_function = lambda label: [1, 0] if label == 1 else [0, 1]

# 575572972, 
# determinism

for _ in range(30):
    seed = random.randint(0, 1000000000)
    dataset_seed = seed
    utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    # Define input and output layers
    input1 = layers.Input(shape=(8, 32, 1)) # fft data
    input2 = layers.Input(shape=(32, 8, 1)) # timeseries
    conv1 = layers.Conv2D(64, (3, 3), activation='relu')(input1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(input2)
    flat1 = layers.Flatten()(conv1)
    flat2 = layers.Flatten()(conv2)
    dense1 = layers.Dense(128, activation='relu')(flat1)
    dense2 = layers.Dense(128, activation='relu')(flat2)
    dense2_2 = layers.Dense(32, activation='relu')(dense2)
    concatenated = layers.Concatenate()([dense1, dense2_2])
    dense = layers.Dense(32, activation='relu')(concatenated)
    output = layers.Dense(2, activation='sigmoid')(dense)

    model = models.Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # execution
    for dataset_path in [os.path.join(datasets_path, subdir) for subdir in datasets_directories]:

        train_data, excluded_datasets = mybci.dataset_loading.load_all_datasets(dataset_path, exclude_matching=exclude_names, load_excluded=True, verbose=True)
        (x_train, y_train), (x_valid, y_valid) = mybci.dataset_loading.split_dataset_to_xy(
            train_data, 
            xlabels=["fftdata", "timeseries"], 
            label_func=label_function, 
            split=train_valid_split,
            seed=dataset_seed
        )

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))
        # model.save(f"models/model_18n20_06_{epochs}epoch{seed}seed.keras")

        target_dataset = None
        for name, dataset in excluded_datasets.items():
            if "test" not in name:
                continue
            target_dataset = dataset

        # split the datasets without shuffling (so that entries are split timewise)
        calib_dataset, test_dataset = mybci.dataset_loading.split_dataset(target_dataset, split=0.3, preshuffle=False)

        (x, y), (x_valid, y_valid) = mybci.dataset_loading.split_dataset_to_xy(
            calib_dataset,
            xlabels=["fftdata", "timeseries"],
            label_func=label_function,
            split=0.8,
            seed=seed
        )

        (x_test_left, y_test_left), (_, _) = mybci.dataset_loading.split_dataset_to_xy(
            test_dataset[1],
            grouped=False,
            xlabels=["fftdata", "timeseries"],
            label_func=label_function,
            split=0.0,
            seed=seed
        )    
        (x_test_right, y_test_right), (_, _) = mybci.dataset_loading.split_dataset_to_xy(
            test_dataset[2],
            grouped=False,
            xlabels=["fftdata", "timeseries"],
            label_func=label_function,
            split=0.0,
            seed=seed
        )    
        # test for predicting LEFT
        # print("Evaluating LEFT actions before calibration")
        r1 = model.evaluate(x_test_left, y_test_left, return_dict=True)
        # my_evaluate(x_test_left, y_test_left, model)
        # test for predicting RIGHT
        # print("Evaluating RIGHT actions before calibration")
        r2 = model.evaluate(x_test_right, y_test_right, return_dict=True)
        # my_evaluate(x_test_right, y_test_right, model)

        # print(f"Calibrating on {len(y)} samples. Time to collect calibration data: {len(y)*32/255:.2f} seconds.")
        model.fit(x, y, batch_size=batch_size_calib, epochs=epochs_calib, validation_data=(x_valid, y_valid))

        # test for predicting LEFT
        # print("Evaluating LEFT actions")
        r3 = model.evaluate(x_test_left, y_test_left, return_dict=True)
        r3p = my_evaluate(x_test_left, y_test_left, model)

        # test for predicting RIGHT
        # print("Evaluating RIGHT actions")
        r4 = model.evaluate(x_test_right, y_test_right, return_dict=True)
        r4p = my_evaluate(x_test_right, y_test_right, model)

        print("SEED:", seed)

        data_to_save = {"seed": seed, "r1": r1, "r2": r2, "r1calib": r3, "r2calib": r4, "r1calib_my_eval": r3p, "r2calib_my_eval": r4p}
        # print(data_to_save)

        with open(f"seeded_results_0807_myeval44_1/{seed}_epochs{epochs}_epochscalib{epochs_calib}.json", 'w') as f:
            json.dump(data_to_save, f, sort_keys=False, indent=2)
