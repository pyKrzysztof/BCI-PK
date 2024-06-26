import os
import sys
import numpy as np
import pickle


training_data_dir = "data/processed_session_data/chunks_processed_constant_detrend"
training_data = []

# load training data
for filename in os.listdir(training_data_dir):
    # skip fft data, it will be loaded manually
    if not filename.endswith("timedata.csv"):
        continue

    idx1, idx2, _ = filename.split("_")
    fft_filename = f"{idx1}_{idx2}_fftdata.csv"
    timedata = np.genfromtxt(os.path.join(training_data_dir, filename), delimiter=',')
    fftdata = np.genfromtxt(os.path.join(training_data_dir, fft_filename), delimiter=',')

    # reject data with crazy accel
    # not implemented yet

    # remove accel channels
    timedata = timedata[:, :8]

    # remove frequency scale
    fftdata = fftdata[1:, :]

    # get label
    print(filename)
    label = 0 if idx1[0] == "L" else 1

    training_data.append((timedata, fftdata, label))


new_filename = training_data_dir[:-1] + ".pickle"
print(new_filename)
with open(new_filename, 'wb') as f:
    pickle.dump(training_data, f)

# with open(new_filename, 'rb') as f:
#     data = pickle.load(f)
#     print(len(data))
