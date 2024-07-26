import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from keras import models
import random


training_file = "data/lewo_prawo_32_128_const/training1.pickle"
model_name = "base/32_128_model_2.keras"

models_path = "models/"
model_path = os.path.join(models_path, model_name)
my_model : models.Model = models.load_model(model_path)

training_data = None
with open(training_file, 'rb') as f:
    training_data = pickle.load(f)

random.shuffle(training_data)
test_data = training_data[-200:]
training_data = training_data[:-200]

datax1 = np.array([x1 for x1, _, _ in training_data])
datax2 = np.array([x2 for _, x2, _ in training_data])
Y = np.array([[not y, y] for _, _, y in training_data])

print(f"training data: {len(datax1)}")

my_model.fit([datax2, datax1], Y, batch_size=32, epochs=10, validation_split=0.2)

datax1_test = np.array([x1 for x1, _, _ in test_data])
datax2_test = np.array([x2 for _, x2, _ in test_data])
Y_test = np.array([[not y, y] for _, _, y in test_data])

print(f"test data: {len(datax1_test)}")

loss, accuracy = my_model.evaluate([datax2_test, datax1_test], Y_test)
print(f"loss: {loss}\naccuracy: {accuracy}")

my_model.save(os.path.join(models_path, model_name.split("/")[1]))
