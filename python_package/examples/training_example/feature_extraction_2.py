import os
import mybci
import mybci.feature_extraction

from mybci.feature_extraction import Filters, Features

###################################################### FILTERS and FEATURE PREPARATION



def preprocess():
    config = mybci.get_base_config()
    config['name'] = "processed_data"
    config['action_markers'] = [1, 2]
    config['session_file'] = []
    # config['session_file'].extend(['data/session/session_data/2805.csv'])
    # config['session_file'].extend([f'data/session_data/0406_{i}.csv' for i in [1, 2]])
    config['session_file'].extend([f'data/session_data/1806_{i}.csv' for i in range(1, 6)])
    config['session_file'].extend([f'data/session_data/2006_{i}.csv' for i in range(1, 6)])
    config['session_file'].extend(['data/session_data/calib_data.csv'])
    config['packet_size'] = 128 # 64
    config['filter_size'] = 128 # 64
    config['filter_func'] = {f'f1': mybci.feature_extraction.get_data_filter(
        filters=[Filters.bandpass(1.0, 45.0, 3), Filters.bandstop(49, 51, 4), Filters.wavelet_denoise(3)], 
        value_scale=1e-6
    )}

    config['ml_prepare_size'] = 128
    config['ml_prepare_func'] = {'p1': mybci.feature_extraction.get_feature_extractor(
        features=[Features.WAVELETS, Features.BANDPOWERS]
    )}

    config['save_training_dataset'] = True
    config['keep_seperate_training_data'] = False
    config['output_path_training_dataset'] = "data/datasets/"

    processor = mybci.DataProcessor(config)
    processor.process()

######################################################  MODEL CREATION

# preprocess()


from keras import models, layers, utils

epochs = 5
batch_size = 64
batch_size_calib = 16
epochs_calib = 3

calib_split = 0.6

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

datasets_path = "data/datasets/processed_data/"
datasets_directories = [f"f{i}_p1" for i in [1]]
exclude_names = ["2004", "0406", "2006_5", "calib", "test"]


dataset_seed = None
for dataset_path in [os.path.join(datasets_path, subdir) for subdir in datasets_directories]:

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

    print(f"Calibrating on {len(y)} samples. Time to collect calibration data: {len(y)*64/250:.2f} seconds.")
    model.fit(x, y, batch_size=batch_size_calib, epochs=epochs_calib, validation_data=(x_valid, y_valid))

    # test for predicting LEFT
    print("Evaluating LEFT actions")
    model.evaluate(x_test_left, y_test_left)
    # my_evaluate(x_test_left, y_test_left, model)

    # test for predicting RIGHT
    print("Evaluating RIGHT actions")
    model.evaluate(x_test_right, y_test_right)
    # my_evaluate(x_test_right, y_test_right, model)