import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from keras import models
import random


def load(path):
    data = None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

training_file_og = "data/processed_session_data/chunks_processed_constant_detrend/training.pickle"
model_name = "base/32_128_model_2.keras"

models_path = "models/"
model_path = os.path.join(models_path, model_name)
my_model : models.Model = models.load_model(model_path)

training_data_og = load(training_file_og)
training_data_1 = load("data/chunks_processed/1/trainin.pickle")
training_data_2 = load("data/chunks_processed/2/trainin.pickle")
random.shuffle(training_data_og)
random.shuffle(training_data_1)
random.shuffle(training_data_2)
# training_data = training_data_1 + training_data_2 + training_data_og
# training_data = training_data_og

test_data = training_data_1[-200:] + training_data_2[-200:]
test_data_old = training_data_og[-200:]
training_data = training_data_1[:-200] + training_data_2[:-200] + training_data_og[:-200]



datax1 = np.array([x1 for x1, _, _ in training_data])
datax2 = np.array([x2 for _, x2, _ in training_data])
Y = np.array([[not y, y] for _, _, y in training_data])
print(sum(Y), len(Y))

print(f"training data: {len(datax1)}")

my_model.fit([datax2, datax1], Y, batch_size=32, epochs=10, validation_split=0.2)
my_model.save(os.path.join(models_path, model_name.split("/")[1]))


datax1_test = np.array([x1 for x1, _, _ in test_data])
datax2_test = np.array([x2 for _, x2, _ in test_data])
Y_test = np.array([[not y, y] for _, _, y in test_data])

print(f"test data new : {len(datax1_test)}")

loss, accuracy = my_model.evaluate([datax2_test, datax1_test], Y_test)
print(f"loss: {loss}\naccuracy: {accuracy}")



datax1_test = np.array([x1 for x1, _, _ in test_data_old])
datax2_test = np.array([x2 for _, x2, _ in test_data_old])
Y_test = np.array([[not y, y] for _, _, y in test_data_old])

print(f"test data old: {len(datax1_test)}")

loss, accuracy = my_model.evaluate([datax2_test, datax1_test], Y_test)
print(f"loss: {loss}\naccuracy: {accuracy}")

