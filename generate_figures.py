import os
import pickle
import sys
import logging
import matplotlib.pyplot as plt
import params_and_cli as params
import numpy as np
from tqdm import tqdm

logging.disable()
path = '/'.join(os.path.realpath(__file__).split('/')[:-1])


params.parse_wrapper(sys.argv[1:], 'figures')


if 'dataset' in os.listdir(path):
    all_results = [file for file in os.listdir(f'{path}/dataset') if '.pickle' in file]
    params.params['figures']['subplots-number'] = min(params.params['figures']['subplots-number'], len(all_results))
    rnd_indexes = np.random.randint(0, len(all_results), params.params['figures']['subplots-number'])
else:
    print('\n./dataset directory not found. please run generate_data.py first.')
    exit()


if 'figures' in os.listdir(path):
    os.system(f'rm -rf {path}/figures/*')
else:
    os.makedirs(f'{path}/figures')


params.print_script_description('figures')
for iter_idx, rnd_idx in enumerate(tqdm(rnd_indexes, desc='Generating figures in ./figures')):
    result_file = all_results[rnd_idx]
    with open(f'{path}/dataset/{result_file}', 'rb') as f:
        tracer_result = pickle.load(f)

    tracer_img = tracer_result.native

    if iter_idx == 0:
        fig, ax = plt.subplots(1, 1, figsize=tracer_img.shape, dpi=1)
        ax.set_position([0, 0, 1, 1])

    
    ax.imshow(tracer_img, cmap=params.params['figures']['cmap'])
    ax.axis('off')
    
    fig.savefig(f'{path}/figures/img{iter_idx+1}.jpg')
plt.close(fig)
os.system(f'rm ./root.log')
