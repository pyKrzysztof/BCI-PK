import numpy as np
import tensorflow as tf
from keras import models, utils
from brainflow import DataFilter, BoardShim, BoardIds
import pickle


model = "training/models/32_128_model_2.keras"

my_model : models.Model = models.load_model(model)

datafile = "training/data/zero_data.pickle"
data = None
with open(datafile, 'rb') as f:
    data = pickle.load(f)


X1 = np.array([x for x, _, _ in data])
X2 = np.array([x for _, x, _ in data])

for x1, x2 in zip(X1, X2):
    reshaped_1 = x1.reshape(1, 32, 8, 1)
    reshaped_2 = x2.reshape(1, 8, 32, 1)
    value = my_model.predict([reshaped_2, reshaped_1])
