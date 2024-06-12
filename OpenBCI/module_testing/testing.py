import mybci
import pandas as pd
import random as rd
import numpy as np

from keras import models
from brainflow.data_filter import DataFilter, WindowOperations













def chunk_func(file_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    N = 32
    X = 128
    SKIP = 250
    sep = "\t"
    # Load the CSV file
    df = pd.read_csv(file_path, header=None, sep=sep)

    # List to store the results
    result = []

    # Iterate through the DataFrame in steps of N+X
    i = SKIP
    while i < len(df):
        # Skip N rows
        end_index = i + N

        # Ensure no out of bounds
        start_index = end_index - X
        if end_index > len(df):
            break

        # Take the last X rows
        chunk = df.iloc[start_index:end_index]

        # Process the data
        timeseries_data = chunk.iloc[-N:]
        timeseries_data = timeseries_data[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        chunk_np = chunk.to_numpy().transpose()

        data = [[], ]
        for channel in range(1, 9):
            fft_data = DataFilter.get_psd_welch(chunk_np[channel], X, X // 2, SKIP, WindowOperations.BLACKMAN_HARRIS)
            lim = min(32, len(fft_data[0]))
            values = fft_data[0][0:lim].tolist()
            data[0] = fft_data[1][0:lim].tolist()
            data.append(values)

        fft = np.array(data)
        fft = pd.DataFrame(fft)

        result.append({'timedata':timeseries_data, 'fftdata':fft})

        # Move the index forward
        i = end_index

    return result

mybci.process_chunks("data/chunks/0406_1/", "data/training/0406_1/data/", chunk_func, "\t")
mybci.process_chunks("data/chunks/0406_2/", "data/training/0406_2/data/", chunk_func, "\t")
mybci.process_chunks("data/chunks/2805/", "data/training/2805/data/", chunk_func, "\t")


# removes freq axis from fftdata and accel data from timedata.
funcs = {"timedata": lambda data: data[:, :8], "fftdata": lambda data: data[1:, :]}

input_dir = "data/training/0406_1/data/"
output_file = "data/training/0406_1.pickle"
mybci.create_training_data(input_dir, output_file, [1, 2], funcs, "\t")
train_data_1, test_data_1 = mybci.load_and_split_data(output_file, 0.2, load_all=False)

input_dir = "data/training/0406_2/data/"
output_file = "data/training/0406_2.pickle"
mybci.create_training_data(input_dir, output_file, [1, 2], funcs, "\t")
train_data_2, test_data_2 = mybci.load_and_split_data(output_file, 0.2, load_all=False)

input_dir = "data/training/2805/data/"
output_file = "data/training/2805.pickle"
mybci.create_training_data(input_dir, output_file, [1, 2], funcs, "\t")
train_data_3, test_data_3 = mybci.load_and_split_data(output_file, 0.2, load_all=False)


# training
train_data = []
train_data.extend(train_data_1)
train_data.extend(train_data_2)
train_data.extend(train_data_3)
rd.shuffle(train_data)
model : models.Model = models.load_model("models/base/32_128_model_2.keras")
X1 = np.array([data['fftdata'] for data in train_data])
X2 = np.array([data['timedata'] for data in train_data])
Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in train_data])
model.fit([X1, X2], Y, batch_size=16, epochs=300, validation_split=0.2)

# testing
def test_count(data, idx_zero, idx_non_zero):
    test = [0, 0]
    for a, b in data:
        test[0] = test[0] + a
        test[1] = test[1] + b
    assert test[idx_zero] == 0
    assert test[idx_non_zero] != 0

X1 = np.array([data['fftdata'] for data in test_data_1[1]])
X2 = np.array([data['timedata'] for data in test_data_1[1]])
Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data_1[1]])
test_count(Y, 1, 0)
model.evaluate([X1, X2], Y)

X1 = np.array([data['fftdata'] for data in test_data_1[2]])
X2 = np.array([data['timedata'] for data in test_data_1[2]])
Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data_1[2]])
test_count(Y, 0, 1)
model.evaluate([X1, X2], Y)

X1 = np.array([data['fftdata'] for data in test_data_2[1]])
X2 = np.array([data['timedata'] for data in test_data_2[1]])
Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data_2[1]])
test_count(Y, 1, 0)
model.evaluate([X1, X2], Y)

X1 = np.array([data['fftdata'] for data in test_data_2[2]])
X2 = np.array([data['timedata'] for data in test_data_2[2]])
Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data_2[2]])
test_count(Y, 0, 1)
model.evaluate([X1, X2], Y)

X1 = np.array([data['fftdata'] for data in test_data_3[1]])
X2 = np.array([data['timedata'] for data in test_data_3[1]])
Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data_3[1]])
test_count(Y, 1, 0)
model.evaluate([X1, X2], Y)

X1 = np.array([data['fftdata'] for data in test_data_3[2]])
X2 = np.array([data['timedata'] for data in test_data_3[2]])
Y = np.array([[1, 0] if data['label'] == 1 else [0, 1] for data in test_data_3[2]])
test_count(Y, 0, 1)
model.evaluate([X1, X2], Y)
