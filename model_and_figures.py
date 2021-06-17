"""
    This script creates a tensorflow CNN model, trains it, then saves it and produces figures locally in ./figures

    hyperparameters controlling the training and figures can be set using the cli arguments while running the script.
    For example: 
            python model_and_figures.py --epochs=50 -f=png

    To view all the possible argumnets and their description use:
            python model_and_figures.py --help
"""


# import necessary modules
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable debug messages by tensorflow

import pickle
import sys
import logging
logging.disable(logging.CRITICAL)   # disable log messages by autolens

import warnings

import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np
import params_and_cli as params     # import the parameters and cli script
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from numpy.core.defchararray import mod

# supress debug, log, and warning messages to the terminal
warnings.filterwarnings('ignore')   # disable warning messages by matplotlib


# import necessary tensorflow modules


# the absolute path of the script to make sure the data is generated in the project folder not current directory
path = '/'.join(os.path.realpath(__file__).split('/')[:-1])


def main():
    """the main routine of the script
    """

    params.parse_wrapper(sys.argv[1:], 'model-figures') # parse the passed arguments
    # load the data files and split it into train, validation, and test sets
    x, y = load_dataset()                    
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = split_data(x, y)

    # create a CNN model and train it
    model = create_model(input_shape=train_x.shape[1:])
    history = train_model(model, train_x, train_y, valid_x, valid_y)

    # remove old figures and make sure ./figures directory exists
    if 'figures' in os.listdir(path):
        os.system(f'rm -rf {path}/figures/*')
    else:
        os.makedirs(f'{path}/figures')

    # save the model
    print('\n\nSaving model as .h5 file...')
    model.save(f'{path}/lens_cnn.h5')

    # generate the figures to ./figures
    generate_sample_figures(x)
    generate_loss_figure(model, history)
    generate_activation_figure(model, test_x)

    # print the script description for more information
    params.print_script_description('model-figures')


def load_dataset():
    """loads the images from the .pickle files and the labels dataframe

    Returns:
        (np.array, pd.DataFrame): results as images array and labels as a dataframe
    """
    try:
        # try to read the csv file and the .pickle files
        param_df = pd.read_csv(f'{path}/dataset/param_df.csv')
        with open(f'{path}/dataset/{param_df.iloc[0, 0]}', 'rb') as f:
            img_size = pickle.load(f).native.shape[0]   # the image size of the result

        # initialize the data array
        data = np.zeros(shape=(param_df.shape[0], img_size, img_size, 1))

        for idx, file in enumerate(param_df['file']):
            with open(f'{path}/dataset/{file}', 'rb') as f:
                data[idx, :, :, 0] = pickle.load(f).native  # read .pickle files as arrays

    except FileNotFoundError:
        print('\n./dataset directory doesn\'t exist or missing data files. please run generate_data.py first.')
        exit()

    # return only the columns specified as labels for the model
    return data, param_df[params.params['model-figures']['labels']]


def split_data(x, y):
    """splits the data and labels to train, validation, and test sets

    Args:
        x (np.array): the data as lens images in the array
        y (pd.DataFrame): a dataframe with the columns used as labels

    Returns:
        (np.array, pd.DataFrame), (np.array, pd.DataFrame), (np.array, pd.DataFrame): train, validation, and test sets
    """

    all_indexes = np.array(range(x.shape[0]))   # indexes of all the rows
    # randomly select the test indexes
    test_indexes = np.random.choice(all_indexes, int(
        x.shape[0]*params.params['model-figures']['test-split']))

    # the remaining indexes
    all_indexes = np.array(list(set(all_indexes).difference(test_indexes)))

    # randomly select the validation indexes
    valid_indexes = np.random.choice(all_indexes, int(
        all_indexes.size*params.params['model-figures']['valid-split']))

    # remaining indexes as training indexes
    train_indexes = np.array(list(set(all_indexes).difference(valid_indexes)))

    # create the sets
    train_x, train_y = x[train_indexes], y.iloc[train_indexes]
    valid_x, valid_y = x[valid_indexes], y.iloc[valid_indexes]
    test_x, test_y = x[test_indexes], y.iloc[test_indexes]

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def create_model(input_shape):
    """creates the tensorflow CNN model and compiles it

    Args:
        input_shape (tuple): the shape of the input images of the model

    Returns:
        [tf.keras.Sequential]: the CNN tensorflow model class
    """
    # grab the number of labels to predict
    labels_len = len(params.params['model-figures']['labels'])

    # create the model
    model = Sequential(name='Gravitational_Lens_Model')

    # the input layer with proper shape
    model.add(layers.Input(input_shape))

    # a series of convolutional and pooling layers
    model.add(layers.Conv2D(filters=32, kernel_size=(6, 6),
                            strides=(1, 1), padding='same', activation='tanh'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='same'))
    
    model.add(layers.Conv2D(filters=64, kernel_size=(4, 4),
                            strides=(1, 1), padding='same', activation='tanh'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='same'))
    
    model.add(layers.Conv2D(filters=128, kernel_size=(4, 4),
                            strides=(1, 1), padding='same', activation='tanh'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='same'))

    # the last few fully-connected layers at the end of the model
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(labels_len, activation='linear'))

    # compile the model and return it
    model.compile(optimizer=optimizers.Adam(
        params.params['model-figures']['learning-rate']), loss='mse', metrics=['mae'])
    return model


def train_model(model, train_x, train_y, valid_x, valid_y):
    """trains the tensorflow model and returns the log history of the training

    Args:
        model (tf.keras.sequential): the CNN model object
        train_x (np.array): the training data
        train_y (pd.DataFrame): the training labels
        valid_x (np.array): the validation data
        valid_y (pd.DataFrame): the validation labels

    Returns:
        tf.keras.history: the log of the training process with the loss values
    """

    # grab the hyperparameters from the params script
    epochs = params.params['model-figures']['epochs']
    batch_size = params.params['model-figures']['batch-size']

    history = model.fit(train_x, train_y, validation_data=(
        valid_x, valid_y), batch_size=batch_size, epochs=epochs)
    return history


def generate_sample_figures(x):
    """plots the figures of the generated gravitational lens samples and saves them

    Args:
        x (np.array): the gravitational lens data as images array
    """

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # iterate over the first few images (specified by the parameter)
    for iter_idx, tracer_img in enumerate(tqdm(x[:params.params['model-figures']['subplots-number']],
                                               desc='Generating samples figures in ./figures')):
        # plot the image and save it
        ax.imshow(tracer_img, cmap=params.params['model-figures']['cmap'])
        ax.axis('off')
        ax.set_title(
            f'Generated Gravitational Lens Sample #{iter_idx+1}', size=20)
        plt.tight_layout()
        fig.savefig(f'{path}/figures/img{iter_idx+1}.jpg')

        plt.close(fig)
    os.system(f'rm ./root.log')     # remove log files generated by autolens


def generate_loss_figure(model, history):
    """plots the training losses figure and saves it to ./figures

    Args:
        model (tf.keras.Sequential): the CNN model
        history (tf.keras.history): the log of the training process with the loss values
    """

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))  # create the figure with 1 axes

    train_epochs = np.array(history.epoch) + 1      # increment the epochs to start from 1
    colors = ['green', 'red']                       # colors for train and validation losses

    for key, values in history.history.items():
        # modify the loss labels saved by tensorflow to suitable ones for the figure
        label = key if key[-4:] != 'loss' else key[:-4] + model.loss
        label = 'Training ' + label if 'val' not in label else label
        label = label.replace('val_', 'Validation ').replace(
            'mae', 'Mean Absolute Error').replace('mse', 'Mean Squared Error')

        # plot with the proper line style and color depending on loss type and set type
        ax.plot(train_epochs, values, label=label, linestyle=':' if 'Abs' in label else '-', linewidth=2,
                c=colors[int('Val' in label)], marker='o', markersize=6)
    
    # format the axes 
    ax.grid('on')
    ax.set_xticks(train_epochs)
    ax.set_xticklabels(ax.get_xticks(), size=14, rotation=45)
    ax.set_yticklabels(np.round(ax.get_yticks(), 1), size=14)
    ax.set_xlabel('Training Epochs', size=14)
    ax.set_ylabel('Loss Value', size=14)

    ax.legend(loc='upper right', ncol=2, fancybox=True, fontsize=14)
    ax.set_title('Losses vs Epochs', size=24)
    plt.tight_layout()

    fig.savefig(
        f'{path}/figures/loss_epochs.{params.params["model-figures"]["figure-format"]}', bbox_inches='tight')


def generate_activation_figure(model, test_x, layers_num=3):
    """plots the intermediate activations of the convolutional layers of the model and saves them

    Args:
        model (tf.keras.Sequential): the CNN model
        test_x (np.array): the test data as images
        layers_num (int, optional): number of layers to visualize their activations. Defaults to 3.
    """

    # the layers outputs to visualize (defaults to the first 3 conv layers)
    layers_output = [
        layer.output for layer in model.layers if layer.__class__.__name__ == 'Conv2D'][:layers_num]

    # define the first part of the model as a function to grab their output
    functor = K.function([model.input], layers_output)

    # select a random test image
    test_idx = np.random.randint(0, test_x.shape[0])
    test_img = test_x[[test_idx]]

    # plot the input image and save it to ./figures
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(test_img[0])
    ax.axis('off')
    ax.set_title('Input Image to CNN Model', size=24)
    plt.tight_layout()
    fig.savefig(f'{path}/figures/conv_input.jpg')

    # loop over outputs of the function (each layer output)
    for out_idx, conv_output in enumerate(tqdm(functor([test_img]),
                                               desc='Generating CNN activation figures in ./figures')):
        dim_sqrt = np.sqrt(conv_output.shape[-1])
        # get integer number of rows and columns (ceiled values = extra axes subplots)
        rows_num, cols_num = int(
            np.ceil(dim_sqrt)), conv_output.shape[-1] // int(dim_sqrt)

        fig, axs = plt.subplots(rows_num, cols_num, figsize=(6, 6))

        for idx, ax in enumerate(axs.ravel()):      # loop over axes
            if idx < conv_output.shape[-1]:         # if a filter map remains, plot it
                activations = conv_output[0, :, :, idx]
                ax.imshow(activations, cmap='viridis')
            ax.set_axis_off()                       # disable the axis

        fig.suptitle(
            f'Activations of Convolution Layer #{out_idx+1} with {conv_output.shape[-1]} Filters', size=15)
        plt.tight_layout()
        fig.savefig(f'{path}/figures/conv_activ{out_idx+1}.jpg')

    plt.close('all')


if __name__ == '__main__':
    main()
