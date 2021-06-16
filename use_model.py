import sys

from tensorflow.python.keras.backend import dropout

import params_and_cli as params
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.activations as activations
from tensorflow.keras.models import Sequential


params.parse_wrapper(sys.argv[1:], 'model')

def create_model(input_shape):
    labels_len = 0
    for label_name, label_len in params.params['model']['labels']:
        labels_len += label_len

    model = Sequential(name='Gravitational Lens Model')
    model.add(layers.InputLayer(input_shape))
    
    model.add(layers.Conv2D(filters=32, kernel_size=(6, 6), strides=(1, 1), padding='same', activation='tanh'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(6, 6), strides=(1, 1), padding='same', activation='tanh'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(6, 6), strides=(1, 1), padding='same', activation='tanh'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='same'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(labels_len, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    return model


def train_model(model, x, y, valid_split):
    epochs = params.params['model']['epochs']
    batch_size = params.params['model']['batch-size']
    model.train()
    pass