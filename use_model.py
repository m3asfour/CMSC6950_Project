import logging
import matplotlib.pyplot as plt

from numpy.core.defchararray import mod
logging.disable(logging.CRITICAL)   # disable logging to the terminal by the tensorflow package

import sys
import os 
import pickle

import params_and_cli as params
import numpy as np
import pandas as pd

import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Sequential


# the absolute path of the script to make sure the data is generated in the project folder not current directory
path = '/'.join(os.path.realpath(__file__).split('/')[:-1])


def create_model(input_shape):
    labels_len = len(params.params['model']['labels'])

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

    model.compile(optimizer=optimizers.Adam(params.params['model']['learning-rate']), loss='mse', metrics=['mae'])
    return model


def load_dataset():
    param_df = pd.read_csv(f'{path}/dataset/param_df.csv')
    with open(f'{path}/dataset/{param_df.iloc[0, 0]}', 'rb') as f:
        img_size = pickle.load(f).native.shape[0]
    data = np.zeros(shape=(param_df.shape[0], img_size, img_size, 1))

    for idx, file in enumerate(param_df['file']):
        with open(f'{path}/dataset/{file}', 'rb') as f:
                data[idx, :, :, 0] = pickle.load(f).native

    return data, param_df[params.params['model']['labels']]

def split_data(x, y):
    all_indexes = np.array(range(x.shape[0]))
    test_indexes = np.random.choice(all_indexes, int(x.shape[0]*params.params['model']['test-split']))
    
    all_indexes = np.array(list(set(all_indexes).difference(test_indexes)))
    valid_indexes = np.random.choice(all_indexes, int(all_indexes.size*params.params['model']['valid-split']))
    
    train_indexes = np.array(list(set(all_indexes).difference(valid_indexes)))

    train_x, train_y = x[train_indexes], y.iloc[train_indexes]
    valid_x, valid_y = x[valid_indexes], y.iloc[valid_indexes]
    test_x, test_y = x[test_indexes], y.iloc[test_indexes]
    
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def train_model(model, train_x, train_y, valid_x, valid_y, save_log=True):
    epochs = params.params['model']['epochs']
    batch_size = params.params['model']['batch-size']

    history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), batch_size=batch_size, epochs=epochs)
    return history
    

def generate_loss_figure(model, history):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    train_epochs = np.array(history.epoch) + 1
    colors = ['green', 'red']
    for key, values in history.history.items():
        label = key if key[-4:] != 'loss' else key[:-4] + model.loss
        label = 'Training ' + label if 'val' not in label else label
        label = label.replace('val_', 'Validation ').replace('mae', 'Mean Absolute Error').replace('mse', 'Mean Squared Error')
        ax.plot(train_epochs, values, label=label, linestyle=':' if 'Abs' in label else '-', linewidth=2, 
                c=colors[int('Val' in label)], marker='o', markersize=6)
    ax.grid('on')
    ax.set_xticks(train_epochs)
    # ax.set_yticks(np.linspace(*ax.get_ylim(), 10))
    ax.set_xticklabels(ax.get_xticks(), size=14, rotation=45)
    ax.set_yticklabels(np.round(ax.get_yticks(), 1), size=14)
    ax.set_xlabel('Training Epochs', size=14)
    ax.set_ylabel('Loss Value', size=14)

    ax.legend(loc='upper right', ncol=2, fancybox=True, fontsize=14)
    ax.set_title('Losses vs Epochs', size=24)
    
    fig.savefig(f'{path}/figures/loss_epochs.jpg', bbox_inches='tight')
    return ax


def main():
    params.parse_wrapper(sys.argv[1:], 'model')
    x, y = load_dataset()
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = split_data(x, y)

    model = create_model(input_shape=train_x.shape[1:])
    history = train_model(model, train_x, train_y, valid_x, valid_y)

    model.save(f'{path}/lens_cnn.h5')
    return generate_loss_figure(model, history)

if __name__ == '__main__':
    main()
