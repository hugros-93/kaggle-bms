import numpy as np
import pandas as pd
import random

import plotly.graph_objects as go

from tensorflow.keras import Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.layers import (
    InputLayer,
    Dense,
    Flatten,
    Reshape,
    RepeatVector,
    Conv2D,
    GRU,
    BatchNormalization
)


def loss_function(labels, logits):
    '''Adapted loss function'''
    return sparse_categorical_crossentropy(labels, logits, from_logits=True)


def plot_history(history):
    '''Plot the train and test loss function for each epoch'''
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['loss'], name='training loss'))
    fig.add_trace(go.Scatter(
        y=history.history['val_loss'], name='validation loss'))

    fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Loss",
        title="Training history"
    )
    return fig
