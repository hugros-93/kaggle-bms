import os
import cv2
import numpy as np
import random
import plotly.express as px
import matplotlib.pyplot as plt
from Levenshtein import distance
import tensorflow as tf

def show_image(X):
    fig = px.imshow(X, binary_string=True)
    fig.update_layout(
        coloraxis_showscale=False, margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

class ImageSetObject:
    def __init__(self, list_names, list_paths):
        self.list_names = list_names
        self.list_paths = list_paths
        self.X = []
        self.image_ids = [x.split('.')[0] for x in self.list_names]

    def prepare_data(self, new_shape, filtering=True, adjust=True):
        for i, name in enumerate(self.list_names):
            Image = ImageObject(name, self.list_paths[i])
            Image.load_picture()
            if filtering: Image.filter()
            if adjust: Image.adjust()
            Image.resize(new_shape)
            self.X.append(Image.X)
        self.X = np.array(self.X)
        self.shape = self.X.shape

class ImageObject:
    def __init__(self, name, path):
        self.name = name
        self.path = path

    def load_picture(self):
        self.X = plt.imread(f"{self.path}{self.name}")
        self.X = np.array(self.X)
        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], 1))
        self.X = 1.0 - self.X
        self.shape = self.X.shape

    def filter(self, d=2, r=0.15):
        X = self.X
        new_X = X.copy()
        for i in range(d, self.shape[0]-d+1, 2*d):
            for j in range(d, self.shape[1]-d+1, 2*d):
                x = X[i-d:i+d, j-d:j+d].ravel()
                if len(x[x == 1.0]) < r*(2*d)**2:
                    new_X[i-d:i+d, j-d:j+d] = 0.0
        self.X = new_X

    def adjust(self):
        new_x = self.X[:, np.where(self.X.sum(axis=0) > 0)[0], :]
        new_x = new_x[np.where(new_x.sum(axis=1) > 0)[0], :, :]
        self.X = new_x
        self.shape = self.X.shape

    def resize(self, new_dim):
        self.X = tf.image.resize(self.X, new_dim, preserve_aspect_ratio=False).numpy()
        self.X[self.X > 0] = 1.0
        self.shape = self.X.shape