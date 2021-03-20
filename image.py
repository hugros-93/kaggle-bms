import os
import cv2
import numpy as np
import random
import plotly.express as px
import matplotlib.pyplot as plt
from tqdm import tqdm


def pad_image(X, new_shape):
    ht, wd = X.shape
    hh = new_shape[0]
    ww = new_shape[1]
    new_X = np.full((hh, ww), 0.0)
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    new_X[yy:yy+ht, xx:xx+wd] = X
    return new_X


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
        self.list_images = []
        self.image_ids = [x.split('.')[0] for x in self.list_names]

    def load_set(self):
        for i, name in tqdm(enumerate(self.list_names)):
            Image = ImageObject(name, self.list_paths[i])
            Image.load_picture()
            Image.inverse()
            Image.rotate()
            self.list_images.append(Image.X)

    def resize_images(self):
        new_shape = [max([x.shape[0] for x in self.list_images]),
                     max([x.shape[1] for x in self.list_images])]
        self.list_images = [pad_image(x, new_shape) for x in self.list_images]
        self.array = np.array(self.list_images)
        self.shape = self.array.shape


class ImageObject:
    def __init__(self, name, path):
        self.name = name
        self.path = path

    def load_picture(self):
        self.X = plt.imread(f"{self.path}{self.name}")
        self.shape = self.X.shape

    def inverse(self):
        self.X = 1 - self.X

    def rotate(self):
        if self.shape[0] > self.shape[1]:
            self.X = np.swapaxes(self.X, 0, 1)
            self.shape = self.X.shape

    def show(self):
        fig = px.imshow(self.X, binary_string=True)
        fig.update_layout(
            coloraxis_showscale=False, margin={"l": 0, "r": 0, "t": 0, "b": 0}
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.show()

    def filter(self, d=2, r=0.15):
        X = self.X
        new_X = X.copy()
        for i in range(d, self.shape[0]-d+1, 2*d):
            for j in range(d, self.shape[1]-d+1, 2*d):
                x = X[i-d:i+d, j-d:j+d].ravel()
                if len(x[x == 1.0]) < r*(2*d)**2:
                    new_X[i-d:i+d, j-d:j+d] = 0.0
        self.X = new_X

    def resize(self, new_shape):
        new_X = cv2.resize(
            self.X, dsize=(new_shape[0], new_shape[1]
                           ), interpolation=cv2.INTER_CUBIC
        )
        self.X = np.array(new_X)
        self.shape = self.X.shape
