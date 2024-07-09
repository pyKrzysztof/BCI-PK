import os
import mybci
import mybci.recording

from brainflow import BoardIds
from preprocess_functions import filter1, filter2, filter3, filter4, prep1, prep2 # type: ignore (vs code won't resolve this import)


from keras import models

batch_size = 32
epochs = 3
model_path = "models/model_18n20_06_1epoch.keras"

# model = models.load_model(model_path)

datafile_name = mybci.recording.record_data(BoardIds.CYTON_BOARD.value, "/dev/ttyUSB0", 32, pipeline="pipelines/pipeline_calib_20.json", output_name="calib_data.csv", simulated=False)

config = mybci.get_base_config()
config["session_file"] = [datafile_name, ]
config["action_markers"] = [1, 2]
config["packet_size"] = 32
config["filter_size"] = 64
config["filter_func"] = {f"f{i+1}": filter for i, filter in enumerate([filter1])}
config["ml_prepare_size"] = 128
config["ml_prepare_func"] = {"p1": prep1}
config["save_training_dataset"] = True
config["keep_seperate_training_data"] = False
config["output_path_training_dataset"] = "calibration_data/"
processor = mybci.DataProcessor(config)
processor.process()

calib_dataset_path = "calibration_data/f1_p1/"
label_function = lambda label: [1, 0] if label == 1 else [0, 1]

train_data = mybci.dataset_loading.load_all_datasets(calib_dataset_path, verbose=True)
(x_train, y_train), (x_valid, y_valid) = mybci.dataset_loading.split_dataset_to_xy(
    train_data, 
    xlabels=["fftdata", "timeseries"], 
    label_func=label_function, 
    split=0.2,
    seed=2
)

print(len(y_train))

# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))