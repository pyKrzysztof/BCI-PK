import os
import pickle
import random
import mybci.training
import tensorflow as tf
import numpy as np
from keras import models, layers, utils

import mybci

# params
epochs = 5
batch_size = 64

# determinism
dataset_seed = 2
utils.set_random_seed(812)
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

# dataset directories by filter/process functions.
datasets_path = "data/datasets/processed_data/"
datasets_directories = [f"f{i}_p1" for i in [1]]
exclude_names = ["2004", "0406", "2006_5"]


# functions
label_function = lambda label: [1, 0] if label == 1 else [0, 1]

def my_evaluate(X, Y, model):
    count_pos = 0
    count_neg = 0
    add_count_1 = 0
    add_count_2 = 0
    for indices, ys, results in mybci.training.custom_evaluate(X, Y, model, batch_size=4, step=4, return_indexed=True):
        sum_result = 0
        additive_result = np.array([0, 0], dtype=np.float64)

        for i, y, result in zip(indices, ys, results):
            additive_result += result*[1.1, 1.]
            sum_result += sum(result*[1.1, -1.])
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



# execution
for dataset_path in [os.path.join(datasets_path, subdir) for subdir in datasets_directories]:
    
    train_data, excluded_datasets = mybci.dataset_loading.load_all_datasets(dataset_path, exclude_matching=exclude_names, load_excluded=True, verbose=True)
    (x_train, y_train), (x_valid, y_valid) = mybci.dataset_loading.create_training_and_validation_datasets(
        train_data, 
        xlabels=["fftdata", "timeseries"], 
        label_func=label_function, 
        split=0.2,
        seed=dataset_seed
    )

    model = models.Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))

    for name, dataset in excluded_datasets.items():
        if "2006_5" not in name:
            continue
        
        # test for predicting LEFT
        x_test, y_test = mybci.dataset_loading.dataset_to_x_y(dataset[1], ["fftdata", "timeseries"], label_function)
        model.evaluate(x_test, y_test)
        my_evaluate(x_test, y_test, model)


        # test for predicting RIGHT
        x_test, y_test = mybci.dataset_loading.dataset_to_x_y(dataset[2], ["fftdata", "timeseries"], label_function)
        model.evaluate(x_test, y_test)
        my_evaluate(x_test, y_test, model)
