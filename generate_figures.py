import os
import pickle
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


images_num = 6
cmap = 'magma'

if len(sys.argv) == 2:
    if sys.argv[1] == '--help':
        print('\nuse "-n=" to specify the number of subplots in figures. For example "python generate_figures -n=6"')
    elif '-n=' in sys.argv[1]:
        try:
            images_num = int(sys.argv[1].split('=')[-1])
        except ValueError as error:
            print('\nPassed value couldn\'t be converted to integer')
            print(error)
            exit()
    else:
        print('argument not recognized. Please try "python generate figures --help"')
else:
    print('this script requires one parameter exactly. Please try "python generate figures --help" for more information.')

if 'dataset' in os.listdir():
    all_results = os.listdir('dataset')
    all_results.remove('gen_df.csv')
    rnd_results = [all_results[rnd_idx] for rnd_idx in np.random.randint(0, len(all_results), images_num)]
else:
    print('./dataset directory not found. please run generate_data.py first.')
    exit()


for iter_idx, result_file in enumerate(rnd_results):
    with open(f'dataset/{result_file}', 'rb') as f:
        tracer_result = pickle.loads(f)

    tracer_img = tracer_result.native

    if iter_idx == 0:
        fig, ax = plt.subplots(1, 1, figsize=tracer_img.shape, dpi=1)
        ax.set_position([0, 0, 1, 1])

    
    ax.imshow(tracer_img, cmap=params.gen_params['cmap'])
    ax.axis('off')
    
    fig.savefig(f'./dataset/img{i+1}.jpg')
        plt.close(fig)
