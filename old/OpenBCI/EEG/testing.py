import mybci
import random as rd
import numpy as np

from keras import models



""" File structure:

.data/
    session/ - raw session data - input directory
    processed/ - processed session data - first stage
    chunks/ - processed chunks - second stage
    training/ - output directory
        data/ - separate packet data files
        x.pickle - serialized action x training data
        y.pickle - serialized action y training data
        ...
    process_functions/ - processing scripts directory - configuration

"""


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
output_dir = "data/training/2805.pickle"
mybci.create_training_data(input_dir, output_file, [1, 2], funcs, "\t")
train_data_3, test_data_3 = mybci.load_and_split_data(output_file, 0.2, load_all=False)


train_data = []
train_data.extend(train_data_1)
train_data.extend(train_data_2)
train_data.extend(train_data_3)
rd.shuffle(train_data)

model : models.Model = models.load_model("models/base/32_128_model_2.keras")

X1 = np.array([data['fftdata'] for data in train_data])
X2 = np.array([data['timedata'] for data in train_data])
Y = np.array([data['label'] for data in train_data])

model.fit([X1, X2], Y, batch_size=32, epochs=300, validation_split=0.2)

