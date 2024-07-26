import numpy as np
import random as rd

from keras import models
from lib import load_data_set

train_data_dict_0, test_data_dict_0 = load_data_set("2805", split=0.3)
train_data_dict_1, test_data_dict_1 = load_data_set("0406_1", split=0.3)
train_data_dict_2, test_data_dict_2 = load_data_set("0406_2", split=0.3)

model : models.Model = models.load_model("models/base/32_128_model_2.keras")

# train:
train_data = train_data_dict_0['1'] + train_data_dict_0['2'] + \
            train_data_dict_1['1'] + train_data_dict_1['2'] + \
            train_data_dict_2['1'] + train_data_dict_2['2']
rd.shuffle(train_data)

X1 = np.array([x1 for x1, _, _ in train_data])
X2 = np.array([x2 for _, x2, _ in train_data])
Y = np.array([(1, 0) if y == 1 else (0, 1) for _, _, y in train_data])
# print(Y)
model.fit([X2, X1], Y, batch_size=32, epochs=300, validation_split=0.2)


# test:
test_data_0_1 = test_data_dict_0['1']
test_data_0_2 = test_data_dict_0['2']
test_data_1 = test_data_dict_1['1'] + test_data_dict_2['1']
test_data_2 = test_data_dict_1['2'] + test_data_dict_2['2']

X1 = np.array([x1 for x1, _, _ in test_data_0_1])
X2 = np.array([x2 for _, x2, _ in test_data_0_1])
Y = np.array([(1, 0) if y == 1 else (0, 1) for _, _, y in test_data_0_1])
# print(Y)
print("Evaluating action 1 for old data")
model.evaluate([X2, X1], Y)

X1 = np.array([x1 for x1, _, _ in test_data_0_2])
X2 = np.array([x2 for _, x2, _ in test_data_0_2])
Y = np.array([(1, 0) if y == 1 else (0, 1) for _, _, y in test_data_0_2])
# print(Y)
print("Evaluating action 2 for old data")
model.evaluate([X2, X1], Y)

X1 = np.array([x1 for x1, _, _ in test_data_1])
X2 = np.array([x2 for _, x2, _ in test_data_1])
Y = np.array([(1, 0) if y == 1 else (0, 1) for _, _, y in test_data_1])
# print(Y)
print("Evaluating action 1 for new data")
model.evaluate([X2, X1], Y)

X1 = np.array([x1 for x1, _, _ in test_data_2])
X2 = np.array([x2 for _, x2, _ in test_data_2])
Y = np.array([(1, 0) if y == 1 else (0, 1) for _, _, y in test_data_2])
# print(Y)
print("Evaluating action 2 for new data")
model.evaluate([X2, X1], Y)