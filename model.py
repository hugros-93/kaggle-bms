import numpy as np
import pandas as pd
import random
from Levenshtein import distance

import plotly.graph_objects as go

from Levenshtein import distance
from tensorflow.keras import Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy, categorical_crossentropy, binary_crossentropy
from tensorflow.keras.layers import (
    InputLayer,
    ReLU,
    Dense,
    Average,
    Flatten,
    Reshape,
    RepeatVector,
    Add,
    Conv2D,
    Conv1DTranspose,
    MaxPool1D,
    Average, 
    MaxPool2D,
    LSTM,
    BatchNormalization,
    GlobalAveragePooling2D
)

def get_model(max_len, vocab):
        model = Sequential(
            [
                Conv2D(filters=32, kernel_size=5,
                       strides=(1, 1), activation='relu'),
                MaxPool2D(),
                Conv2D(filters=64, kernel_size=5,
                       strides=(1, 1), activation='relu'),
                MaxPool2D(),
                Conv2D(filters=128, kernel_size=5,
                       strides=(1, 1), activation='relu'),
                MaxPool2D(),
                Conv2D(filters=256, kernel_size=5,
                       strides=(1, 1), activation='relu'),
                MaxPool2D(),
                GlobalAveragePooling2D(),
                Dense(len(vocab)),
                Reshape((1, len(vocab))),
                Conv1DTranspose(filters=len(vocab), kernel_size=9, strides=2, activation='relu'),
                Conv1DTranspose(filters=len(vocab), kernel_size=max_len-(9-1), strides=1, activation='softmax'),
            ]
        )
        model.build(input_shape=(None, None, None, 1))
        return model
    
def get_text_from_predict(model, x, idx2char):
    y = model(x)
    y = [idx2char[x] for x in np.argmax(y, axis = 2)]
    y = [''.join(x) for x in y]
    return y

def score(y_true, y_predict):
     return np.mean([distance(a,b) for a,b in zip(y_true, y_predict)])
    
def loss_function(labels, logits):
    '''Adapted loss function'''
    return sparse_categorical_crossentropy(labels, logits, from_logits=True)

