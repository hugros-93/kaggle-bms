import numpy as np
import pandas as pd
import random
import math

from Levenshtein import distance
import tensorflow as tf
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
    Conv2DTranspose,
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
            # Conv2D(filters=32, kernel_size=5,
            #        strides=(1, 1), activation='relu'),
            # MaxPool2D(),
            # Conv2D(filters=64, kernel_size=5,
            #        strides=(1, 1), activation='relu'),
            # MaxPool2D(),
            # Conv2D(filters=128, kernel_size=5,
            #        strides=(1, 1), activation='relu'),
            # MaxPool2D(),
            # Conv2D(filters=256, kernel_size=5,
            #        strides=(1, 1), activation='relu'),
            # MaxPool2D(),
            Conv2D(filters=len(vocab), kernel_size=5,
                   strides=(1, 1), activation='relu'),
            MaxPool2D(),
            GlobalAveragePooling2D(),
            Reshape((1, len(vocab))),
            # Conv1DTranspose(filters=len(vocab), kernel_size=9,
            #                 strides=2, activation='relu'),
            # Conv1DTranspose(filters=len(vocab), kernel_size=max_len -
            #                 (9-1), strides=1, activation='softmax'),
            Conv1DTranspose(filters=len(vocab), kernel_size=max_len,
                            strides=1, activation='softmax'),
        ]
    )
    model.build(input_shape=(None, None, None, 1))
    return model


def get_text_from_predict(model, x, idx2char):
    y = model(x)
    y = [idx2char[x] for x in np.argmax(y, axis=2)]
    y = [''.join(x) for x in y]
    return y


def score(y_true, y_predict):
    return np.mean([distance(a, b) for a, b in zip(y_true, y_predict)])


def loss_function(labels, logits):
    '''Adapted loss function'''
    return sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Reference: https://www.tensorflow.org/tutorials/generative/cvae


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))


class VAE(tf.keras.Model):
    """
    Class to define a Variational AutoEncoder (VAE).
    """

    def __init__(
        self, name_model, latent_dim, input_shape_tuple,
    ):
        super().__init__()
        self.name_model = name_model
        self.latent_dim = latent_dim
        self.input_shape_tuple = input_shape_tuple
        self.inference_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=(input_shape_tuple[1], input_shape_tuple[2], 1)),
                Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                Flatten(),
                Dense(latent_dim + latent_dim),
            ]
        )

        dim=int(input_shape_tuple[1]/4)
        self.generative_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dim,)),
                Dense(units=dim*dim*32, activation='relu'),
                Reshape(target_shape=(dim, dim, 32)),
                Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )


    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(
            x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent)
        logpz = log_normal_pdf(z, 0.0, 0.0)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits

    def sample(self, eps):
        return self.decode(eps, apply_sigmoid=True)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables), loss

    def _save_network(self):
        # serialize model to JSON
        model_json_encoder = self.inference_net.to_json()
        model_json_decoder = self.generative_net.to_json()
        with open(f"outputs/{self.name_model}_encoder.json", "w") as json_file:
            json_file.write(model_json_encoder)
        with open(f"outputs/{self.name_model}_decoder.json", "w") as json_file:
            json_file.write(model_json_decoder)
        # serialize weights to HDF5
        self.inference_net.save_weights(f"outputs/{self.name_model}_encoder.h5")
        self.generative_net.save_weights(f"outputs/{self.name_model}_decoder.h5")

    def load_model(self, batch_size=None):
        # load json and create model
        json_file = open(f"outputs/{self.name_model}_encoder.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.inference_net = tf.keras.models.model_from_json(loaded_model_json)
        self.inference_net.build(input_shape=self.input_shape_tuple)

        json_file = open(f"outputs/{self.name_model}_decoder.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.generative_net = tf.keras.models.model_from_json(
            loaded_model_json)
        self.generative_net.build(input_shape=(batch_size, self.latent_dim))

        # load weights into new model
        self.inference_net.load_weights(f"outputs/{self.name_model}_encoder.h5")
        self.generative_net.load_weights(f"outputs/{self.name_model}_decoder.h5")
        print("Loaded model from disk")

    def train(
        self,
        optimizer,
        train_dataset,
        epochs,
        batch_size,
    ):
        """
        Train VAE

        :param optimizer: tensorflow optimizer
        :param train_dataset: tensorflow dataset for training
        :param validation_dataset: tensorflow dataset for validation
        :param epochs: number of epochs

        """

        self.batch_size = batch_size
        self.train_elbo = math.inf
        self.nb_features = self.input_shape_tuple[1]

        loss_before = 1e10

        for epoch in range(1, epochs + 1):

            train_loss = tf.keras.metrics.Mean()

            for train_x in train_dataset:
                gradients, loss = self.compute_gradients(train_x)
                apply_gradients(optimizer, gradients, self.trainable_variables)
                train_loss(self.compute_loss(train_x))

            self.train_elbo = train_loss.result()
            self._save_network()
            print(f"\t> {epoch} - {self.train_elbo}")
        return self
