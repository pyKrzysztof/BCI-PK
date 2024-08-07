from keras import layers, models
from mybci.feature_extraction import Features


features = [Features.WAVELETS, ]

description = "Model with 10 dense layers of size 256, with 6 channel wavelet lvl 3 input and 2 outputs."

def get_model() -> tuple[Features, models.Model]:
    model = models.Sequential()
    model.add(layers.Input(shape=(6, 152)))
    model.add(layers.Flatten())
    for _ in range(10):
        model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
