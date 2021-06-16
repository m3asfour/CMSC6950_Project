import sys

from tensorflow.python.ops.gen_batch_ops import batch

import params_and_cli as params
import numpy as np
import pandas as pd

import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential

params.parse_wrapper(sys.argv[1:], 'model')

def create_model(input_shape):
    labels_len = 0
    for label_name, label_len in params.params['model']['labels']:
        labels_len += label_len

    model = Sequential(name='Gravitational_Lens_Model')
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


def load_dataset():
    pass

def partition_data(x, y):
    all_indexes = np.array(range(x.shape[0]))
    test_indexes = np.random.choice(all_indexes, int(x.shape[0]*params.params['model']['test-split']))
    
    all_indexes = np.array(list(set(all_indexes).difference(test_indexes)))
    valid_indexes = np.random.choice(all_indexes, int(all_indexes.size*params.params['model']['valid-split']))
    
    train_indexes = np.array(list(set(all_indexes).difference(valid_indexes)))

    train_x, train_y = x[train_indexes], y[train_indexes]
    valid_x, valid_y = x[valid_indexes], y[valid_indexes]
    test_x, test_y = x[test_indexes], y[test_indexes]
    
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def train_model(model, train_x, train_y, valid_x, valid_y, save_log=True):
    epochs = params.params['model']['epochs']
    batch_size = params.params['model']['batch-size']

    history = model.train(train_x, train_y, validation_data=(valid_x, valid_y), abatch_size=batch_size, epochs=epochs)
    return history
    