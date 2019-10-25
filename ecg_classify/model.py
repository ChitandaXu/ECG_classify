from keras import models
from keras import layers


def build_cnn_model(feature_size, number_of_classes):
    model = models.Sequential()
    model.add(layers.Conv1D(64, 5, activation='relu', input_shape=(feature_size, 1)))
    model.add(layers.Conv1D(64, 5, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(number_of_classes, activation='softmax'))
    return model
