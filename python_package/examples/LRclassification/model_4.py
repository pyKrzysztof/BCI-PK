import os
import numpy as np
import tensorflow as tf
from keras import layers, models, utils


# Reshape data
# data1_reshaped = np.expand_dims(timeseries_data, axis=-1)  # Shape (8, 32, 1)
# data2_reshaped = np.expand_dims(fft_data, axis=-1)  # Shape (32, 8, 1)

# Define input layers
input1 = layers.Input(shape=(8, 32, 1)) # fft data
input2 = layers.Input(shape=(32, 8, 1)) # timeseries

# Define convolutional layers
conv1 = layers.Conv2D(64, (3, 3), activation='relu')(input1)
conv2 = layers.Conv2D(64, (3, 3), activation='relu')(input2)

# Flatten the output
flat1 = layers.Flatten()(conv1)
flat2 = layers.Flatten()(conv2)

# Add dense layers on top of each flattened output
dense1 = layers.Dense(64, activation='relu')(flat1)
dense2 = layers.Dense(64, activation='relu')(flat2)
dense2_2 = layers.Dense(64, activation='relu')(dense2)


# Concatenate dense layers
concatenated = layers.Concatenate()([dense1, dense2_2])

dense = layers.Dense(64, activation='relu')(concatenated)
dense_2 = layers.Dense(64, activation='relu')(dense)

# Final output layer
output = layers.Dense(2, activation='sigmoid')(dense)

# Define the model
model = models.Model(inputs=[input1, input2], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

path = "models/base/"
filename = "model_4"
os.makedirs(path, exist_ok=True)

# utils.plot_model(model, os.path.join(path, filename + ".png"), show_shapes=True)
model.save(os.path.join(path, filename + ".keras"), overwrite=False)
