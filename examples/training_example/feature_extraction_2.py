import os
import mybci
import mybci.feature_extraction

from mybci.feature_extraction import Filters, Features, get_data_filter, get_feature_extractor
from mybci.data_processor import Config
###################################################### FILTERS and FEATURE PREPARATION

def preprocess_new():

    data_filter = get_data_filter(
        filters = [
            Filters.bandpass(1.0, 45.0, 3), 
            Filters.bandstop(49, 51, 4), 
            Filters.wavelet_denoise(3)
        ], 
        value_scale = 1e-6
    )

    feature_extractor = get_feature_extractor(
        features = [Features.WAVELETS],
        value_scale = 1,
        # filters = [
        #     Filters.bandpass(1.0, 45.0, 3), 
        #     Filters.bandstop(49, 51, 4), 
        #     Filters.wavelet_denoise(3)
        # ], 
        sampling = 250
    )

    config = Config()
    config.name = "LRClassification_1"
    config.action_markers = [1, 2]
    config.buffer_size = 128
    config.packet_size = 64
    config.feature_size = 128
    config.chunk_offset = 250
    config.sampling_rate = 250
    config.filter_function = data_filter
    config.feature_function = feature_extractor

    config.session_folders = ['data/session_data/', ]
    config.excluded_session_files = ['2805', '0406', ]
    config.dataset_directory = 'data/datasets/'
    config.keep_helper_files = False

    mybci.DataProcessor(config).process()


######################################################  MODEL CREATION

# preprocess_new()

from keras import models, layers, utils

epochs = 10
batch_size = 64
batch_size_calib = 4
epochs_calib = 3

calib_split = 0.4

def new_model():
    data_shape = (8, 152)
    model = models.Sequential()
    model.add(layers.Input(shape=data_shape))
    model.add(layers.Flatten())
    for _ in range(10):
        model.add(layers.Dense(128, activation="relu"))

    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

label_function = lambda label: [1, 0] if label == 1 else [0, 1]

dataset_path = "data/datasets/LRClassification_1/"
exclude_names = ["2004", "0406", "2006_5", "calib", "test"]


dataset_seed = None


dataset, excluded_datasets = mybci.dataset_loading.load_all_datasets(dataset_path, exclude_matching=exclude_names, load_excluded=True, verbose=True)

train_dataset, test_dataset = mybci.dataset_loading.split_dataset(dataset, split=0.2, preshuffle=True, postshuffle=True, seed=dataset_seed)

(x_train, y_train), (x_valid, y_valid) = mybci.dataset_loading.split_dataset_to_xy(
    train_dataset, 
    xlabels=[Features.WAVELETS], 
    label_func=label_function, 
    split=0.2,
    seed=dataset_seed
)

(x_test, y_test), (_, _) = mybci.dataset_loading.split_dataset_to_xy(
    test_dataset, 
    xlabels=[Features.WAVELETS], 
    label_func=label_function, 
    split=0.0,
    seed=dataset_seed
)

print(len(y_train) + len(y_valid))
print(len(y_test))

# print(x_train[0].shape)
# print(x_train[1].shape)
# print(x_train.shape, y_train.shape)
model = new_model()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))
model.evaluate(x_test, y_test, 32)

for name, data in excluded_datasets.items():
    if "2006_5" not in name:
        continue

    test_dataset, calib_dataset = mybci.dataset_loading.split_dataset(data, calib_split, preshuffle=False, postshuffle=True, seed=dataset_seed)

(_, _), (x_test_left, y_test_left) = mybci.dataset_loading.split_dataset_to_xy(
    test_dataset[1],
    grouped=False,
    xlabels=[Features.WAVELETS], 
    label_func=label_function,
    split=1.0,
    seed=dataset_seed
)

(_, _), (x_test_right, y_test_right) = mybci.dataset_loading.split_dataset_to_xy(
    test_dataset[2],
    grouped=False,
    xlabels=[Features.WAVELETS], 
    label_func=label_function,
    split=1.0,
    seed=dataset_seed
)

(x, y), (x_valid, y_valid) = mybci.dataset_loading.split_dataset_to_xy(
    calib_dataset,
    xlabels=[Features.WAVELETS], 
    label_func=label_function,
    split=0.8,
    seed=dataset_seed
)

# test for predicting LEFT
print("Evaluating LEFT actions before calibration")
model.evaluate(x_test_left, y_test_left)
# my_evaluate(x_test_left, y_test_left, model)

# test for predicting RIGHT
print("Evaluating RIGHT actions before calibration")
model.evaluate(x_test_right, y_test_right)

print(f"Calibrating on {len(y)} samples. Time to collect calibration data: {len(y)*128/250:.2f} seconds.")
model.fit(x, y, batch_size=batch_size_calib, epochs=epochs_calib, validation_data=(x_valid, y_valid))

# test for predicting LEFT
print("Evaluating LEFT actions")
model.evaluate(x_test_left, y_test_left)
# my_evaluate(x_test_left, y_test_left, model)

# test for predicting RIGHT
print("Evaluating RIGHT actions")
model.evaluate(x_test_right, y_test_right)
# my_evaluate(x_test_right, y_test_right, model)