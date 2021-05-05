import numpy as np
import pandas as pd
import random
import math

from Levenshtein import distance
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.losses import (
    sparse_categorical_crossentropy,
    categorical_crossentropy,
    binary_crossentropy,
)
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Reshape,
    Activation,
    Conv2D,
    MaxPool2D,
    LSTM,
    BatchNormalization,
)


def get_predictive_network(max_len, vocab_size):
    predictive_network = Sequential(
        [
            Conv2D(filters=16, kernel_size=5, strides=(2, 2)),
            BatchNormalization(),
            Activation("relu"),
            MaxPool2D(),
            Conv2D(filters=32, kernel_size=5, strides=(1, 1)),
            BatchNormalization(),
            Activation("relu"),
            MaxPool2D(),
            Conv2D(filters=64, kernel_size=5, strides=(1, 1)),
            BatchNormalization(),
            Activation("relu"),
            MaxPool2D(),
            Conv2D(filters=128, kernel_size=5, strides=(1, 1)),
            BatchNormalization(),
            Activation("relu"),
            MaxPool2D(),
            Flatten(),
            Dense(max_len * 8),
            BatchNormalization(),
            Activation("relu"),
            Reshape((max_len, 8)),
            LSTM(vocab_size, return_sequences=True, activation="softmax"),
        ]
    )
    return predictive_network


def loss_function(labels, logits):
    """Adapted loss function"""
    return sparse_categorical_crossentropy(labels, logits, from_logits=False)


def get_text_from_predict(model, x, idx2char):
    y = model(x)
    y = [idx2char[x] for x in np.argmax(y, axis=2)]
    y = ["".join(x) for x in y]
    return y


def score(y_true, y_predict):
    return np.mean([distance(a, b) for a, b in zip(y_true, y_predict)])
