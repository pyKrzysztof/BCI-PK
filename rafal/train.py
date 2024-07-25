import os
import pickle
import random
import mybci.training
import tensorflow as tf
import numpy as np
from keras import models, layers, utils

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



def my_evaluate(X, Y, model):
    count_pos = 0
    count_neg = 0
    add_count_1 = 0
    add_count_2 = 0
    for indices, ys, results in mybci.training.custom_evaluate(X, Y, model, batch_size=1, step=1, return_indexed=True):
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
        # print(ys[0], additive_result, additive_result_y, sum_result)

    result = count_pos/(count_pos+count_neg)*100
    if Y[0].tolist() == [0, 1]:
        result = 100 - result
    print(f"Sum accuracy: {result:.2f}%")

    result = add_count_1/(add_count_1+add_count_2)*100
    if Y[0].tolist() == [0, 1]:
        result = 100 - result
    print(f"Additive accuracy: {result:.2f}%")





# params
epochs = 5
batch_size = 64
train_valid_split = 0.2
epochs_calib = 5
batch_size_calib = 32
calib_split = 0.3
bias_left = 1.0
bias_right = 1.0


def seeded_train(seed, n_states=3):
    # determinism
    print("SEED FOR THIS SESSION IS: ", seed)
    dataset_seed = seed
    utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    # Define input and output layers
    input1 = layers.Input(shape=(8, 32, 1)) # fft data
    conv1 = layers.Conv2D(128, (3, 3), activation='relu')(input1)
    flat1 = layers.Flatten()(conv1)
    # flat1 = layers.Flatten()(input1)
    dense1 = layers.Dense(256, activation='relu')(flat1)

    input2 = layers.Input(shape=(32, 8, 1)) # timeseries
    conv2 = layers.Conv2D(128, (3, 3), activation='relu')(input2)
    flat2 = layers.Flatten()(conv2)
    dense2 = layers.Dense(128, activation='relu')(flat2)
    dense2_2 = layers.Dense(128, activation='relu')(dense2)

    concatenated = layers.Concatenate()([dense1, dense2_2])
    dense = layers.Dense(256, activation='relu')(concatenated)
    output = layers.Dense(n_states, activation='softmax')(dense) # softmax for multiclass apparently

    # dataset directories by filter/process functions.
    datasets_path = "data/datasets/rafal_1/"
    datasets_directories = [f"f{i}_ml1" for i in [1]]
    exclude_names = ["15-07-2024", "03-07", "04-07", "05-07"]

    # functions
    if n_states == 3:
        label_function = lambda label: [1, 0, 0] if label == 1 else [0, 1, 0] if label == 2 else [0, 0, 1]
    else:
        label_function = lambda label: [1, 0] if label == 1 else [0, 1] 

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
            seed=dataset_seed,
            excluded_groups=[3, "S"] if n_states == 2 else []
        )

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))
        # model.save(f"models/model_18n20_06_{epochs}epoch{seed}seed.keras")

        target_dataset = None
        for name, dataset in excluded_datasets.items():
            if "15-07" not in name:
                continue
            target_dataset = dataset
            break

        calib_dataset, test_dataset = mybci.dataset_loading.split_dataset(target_dataset, 1-calib_split, seed=dataset_seed)

        (x_train, y_train), (x_valid, y_valid) = mybci.dataset_loading.split_dataset_to_xy(
            calib_dataset, 
            xlabels=["fftdata", "timeseries"], 
            label_func=label_function, 
            split=train_valid_split,
            seed=dataset_seed,
            excluded_groups=[3, "S"] if n_states == 2 else []
        )

        (_, _), (x_test_left, y_test_left) = mybci.dataset_loading.split_dataset_to_xy(
            test_dataset[1],
            grouped=False,
            xlabels=["fftdata", "timeseries"],
            label_func=label_function,
            split=1.0,
            seed=dataset_seed,
        )

        (_, _), (x_test_right, y_test_right) = mybci.dataset_loading.split_dataset_to_xy(
            test_dataset[2],
            grouped=False,
            xlabels=["fftdata", "timeseries"],
            label_func=label_function,
            split=1.0,
            seed=dataset_seed,
        )

        if n_states == 3:
            (_, _), (x_test_stop, y_test_stop) = mybci.dataset_loading.split_dataset_to_xy(
                test_dataset[3],
                grouped=False,
                xlabels=["fftdata", "timeseries"],
                label_func=label_function,
                split=1.0,
                seed=dataset_seed,
            )

        # test for predicting LEFT
        print("Evaluating LEFT actions before calibration")
        model.evaluate(x_test_left, y_test_left)
        my_evaluate(x_test_left, y_test_left, model)

        # test for predicting RIGHT
        print("Evaluating RIGHT actions before calibration")
        model.evaluate(x_test_right, y_test_right)
        my_evaluate(x_test_right, y_test_right, model)

        if n_states == 3:
            # test for predicting STOP
            print("Evaluating STOP actions before calibration")
            model.evaluate(x_test_stop, y_test_stop)
            my_evaluate(x_test_stop, y_test_stop, model)

        model.fit(x_train, y_train, batch_size=batch_size_calib, epochs=epochs_calib, validation_data=(x_valid, y_valid))

        # test for predicting LEFT
        print("Evaluating LEFT actions after calibration")
        model.evaluate(x_test_left, y_test_left)
        my_evaluate(x_test_left, y_test_left, model)

        # test for predicting RIGHT
        print("Evaluating RIGHT actions after calibration")
        model.evaluate(x_test_right, y_test_right)
        my_evaluate(x_test_right, y_test_right, model)

        if n_states == 3:
            # test for predicting STOP
            print("Evaluating STOP actions after calibration")
            model.evaluate(x_test_stop, y_test_stop)
            my_evaluate(x_test_stop, y_test_stop, model)


for _ in range(3):
    seed = random.randint(0, 1e9)
    seeded_train(seed, n_states=2)



# 915289257
# 436025872

# 938774268 - base bias to the left, 66 - 62 post calibration 