import os
import mybci
import random as rd
import tensorflow as tf

from keras import layers, models, utils
from mybci.dataset_loading import load_all_datasets, split_dataset, split_dataset_to_xy
from mybci.feature_extraction import Features

INPUT_SHAPE = (6, 152)
N_LAYERS = 10
LAYER_SIZE = 128
N_STATES = 3

EPOCHS_NORMAL = 10
EPOCHS_CALIBRATION = 10
BATCH_SIZE_NORMAL = 64
BATCH_SIZE_CALIBRATION = 16
TRAIN_VALID_SPLIT_NORMAL = 0.2
TRAIN_VALID_SPLIT_CALIBRATION = 0.4

CALIBRATION_SPLIT = 0.3


def new_model():
    model = models.Sequential()
    model.add(layers.Input(shape=INPUT_SHAPE))
    model.add(layers.Flatten())
    for _ in range(N_LAYERS):
        model.add(layers.Dense(LAYER_SIZE, activation="relu"))

    model.add(layers.Dense(N_STATES, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if N_STATES == 3:
    label_function = lambda label: [1, 0, 0] if label == 1 else [0, 1, 0] if label == 2 else [0, 0, 1] if label == 3 else None
else:
    label_function = lambda label: [1, 0] if label == 1 else [0, 1]


dataset_path = 'datasets/Latency_Fixed/'
test_target = "Nr3"
excluded_files = [test_target, ]

def train_and_test_model(SEED, exluded_names=None):
    utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    train_dataset, excluded = load_all_datasets(dataset_path, exclude_matching=exluded_names, load_excluded=True, verbose=True)
    target_dataset = None

    for name, dataset in excluded.items():
        if test_target not in name:
            continue
        target_dataset = dataset
        break

    test_dataset, calibration_dataset = split_dataset(target_dataset, CALIBRATION_SPLIT, preshuffle=False, postshuffle=True, seed=SEED)
    
    (x_train, y_train), (x_valid, y_valid) = split_dataset_to_xy(
        train_dataset, 
        xlabels=[Features.WAVELETS, ],
        label_func=label_function, 
        split=TRAIN_VALID_SPLIT_NORMAL,
        seed=SEED,
        excluded_groups=[3, "S"] if N_STATES == 2 else []
    )

    (x_train_calib, y_train_calib), (x_valid_calib, y_valid_calib) = split_dataset_to_xy(
        calibration_dataset, 
        xlabels=[Features.WAVELETS, ],
        label_func=label_function, 
        split=TRAIN_VALID_SPLIT_CALIBRATION,
        seed=SEED,
        excluded_groups=[3, "S"] if N_STATES == 2 else []
    )

    (x_test_left, y_test_left), (_, _) = split_dataset_to_xy(
        test_dataset[1], 
        grouped=False,
        xlabels=[Features.WAVELETS, ],
        label_func=label_function, 
        split=0.0,
        seed=SEED,
        excluded_groups=[3, "S"] if N_STATES == 2 else []
    )
    (x_test_right, y_test_right), (_, _) = split_dataset_to_xy(
        test_dataset[2], 
        grouped=False,
        xlabels=[Features.WAVELETS, ],
        label_func=label_function, 
        split=0.0,
        seed=SEED,
        excluded_groups=[3, "S"] if N_STATES == 2 else []
    )
    (x_test_stop, y_test_stop), (_, _) = split_dataset_to_xy(
        test_dataset[3], 
        grouped=False,
        xlabels=[Features.WAVELETS, ],
        label_func=label_function, 
        split=0.0,
        seed=SEED,
        excluded_groups=[3, "S"] if N_STATES == 2 else []
    )


    model = new_model()
    model.fit(x_train, y_train, batch_size=BATCH_SIZE_NORMAL, epochs=EPOCHS_NORMAL, validation_data=(x_valid, y_valid))

    print("Evaluation before calibration:")
    model.evaluate(x_test_left, y_test_left)
    model.evaluate(x_test_right, y_test_right)
    model.evaluate(x_test_stop, y_test_stop)

    model.fit(x_train_calib, y_train_calib, batch_size=BATCH_SIZE_CALIBRATION, epochs=EPOCHS_CALIBRATION, validation_data=(x_valid_calib, y_valid_calib))
    print("Evaluation after calibration:")
    model.evaluate(x_test_left, y_test_left)
    model.evaluate(x_test_right, y_test_right)
    model.evaluate(x_test_stop, y_test_stop)

for seed in [rd.randint(0, 1e6) for _ in range(10)]:
    print("SEED:", seed)
    train_and_test_model(seed, excluded_files)
