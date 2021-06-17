"""
    This script is used to randomly generate gravitational lens problem settings using hyperparameters, and saves the
    generated results as .pickle files in ./dataset with a csv file containing the labels (properties of the source
    galaxy)

    the hyperparameters controlling the generation process can be set using the cli arguments while running the script.
    For example: 
            python generate_data.py --src-intensity-range=0.3,1.2

    To view all the possible argumnets and their description use:
            python generate_data.py --help
"""

import logging
# disable logging to the terminal by the autolens package
logging.disable(logging.CRITICAL)

# import required modules
import os
import pickle
import sys

import pandas as pd
import params_and_cli as params
import autolens as al               # import the autolens package

from tqdm import tqdm


# the absolute path of the script to make sure the data is generated in the project folder not current directory
path = '/'.join(os.path.realpath(__file__).split('/')[:-1])


def main():
    """the main routine of the script when it's run
    """
    params.parse_wrapper(
        sys.argv[1:], 'data')  # parse the arguments passed through the cli with error handling
    # check if the dataset folder exists and is empty
    check_dataset_dir()

    # create a grid to be used for ray tracing by the autolens module to determine the final intensity values on it
    grid = al.Grid2D.uniform(shape_native=(params.params['data']['grid-size'], params.params['data']['grid-size']),
                             pixel_scales=0.05)
    # generate the dataset in path/dataset using the specified grid size
    generate_data_files(grid)
    # remove log files generated by auto lens
    os.system(f'rm ./report.log ./root.log')


def check_dataset_dir():
    """checks if the path/dataset directory is ready for generating data files, and makes sure it exists and empty

    Raises:
        Warning: raised when a previous dataset is found, and supressed if "--force" argument was used in the cli
    """
    if 'dataset' in os.listdir(path):
        if params.flags['force']:
            os.system(f'rm -rf {path}/dataset/*')
        else:
            sys.tracebacklimit = 0
            print('\n')
            raise Warning(
                'Dataset already exists. Please use "--force" to overwrite it! Exitting...')
    else:
        os.makedirs(path+'/dataset')


def generate_data_files(grid):
    """generates .pickle data files containg the ray tracing result and a dataframe containing the labels in path/dataset

    Args:
        grid (al.Grid2D): a 2d grid from autolens package for the solver to use in the ray tracing
    """

    param_df = pd.DataFrame(data=[], columns=[
                            'file', *params.labels_description.keys()])   # a dataframe with the labels
    # print the desciption of the script functionality to the cli
    params.print_script_description('data')

    # loop over the range of number of images to generate with a progress bar
    for i in tqdm(range(params.params['data']['generated-number']), desc='Generating .pickle files to ./dataset'):
        # get a random sample of properties using the parameters in params_and_cli.py
        sample = params.generate_sample()

        # define the lens galaxy according to the sampled properties
        lens_galaxy = al.Galaxy(
            redshift=sample['lens-redshift'],
            bulge=al.lmp.EllSersic(
                centre=sample['lens-center'],
                elliptical_comps=sample['lens-ellip'],
                intensity=sample['lens-intensity'],
                effective_radius=sample['lens-effect'],
                sersic_index=sample['lens-serisic'],
                mass_to_light_ratio=sample['masslight-ratio'],
            ),
            disk=al.lmp.EllExponential(
                centre=(0.0, 0.0),
                elliptical_comps=(0.0, 0.1),
                intensity=1.0,
                effective_radius=2.0,
                mass_to_light_ratio=0.2,
            ),
            dark=al.mp.SphNFW(centre=(0.0, 0.0),
                              kappa_s=0.08, scale_radius=30.0),
        )

        # define the source galaxy according to the sampled properties
        source_galaxy = al.Galaxy(
            redshift=sample['src-redshift'],
            bulge=al.lp.EllSersic(
                centre=sample['src-center'],
                elliptical_comps=sample['src-ellip'],
                intensity=sample['src-intensity'],
                effective_radius=sample['src-effect'],
                sersic_index=sample['lens-serisic'],
            ),
        )

        # get the result of the gravitational lens by ray tracing
        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
        # get the intensity values from the grid with image coordinates
        tracer_result = tracer.image_2d_from_grid(grid)

        # save the result as a .pickle file
        with open(f'{path}/dataset/tracer_result{i+1}.pickle', 'wb') as f:
            pickle.dump(tracer_result, f)

        # the pandas dataframe corresponding to the file of this iteration [filename, *all_allowed_labels]
        df_row = [f'tracer_result{i+1}.pickle', sample['src-redshift'], *sample['src-center'], *sample['src-ellip'],
                  sample['src-intensity'], sample['src-effect'], sample['lens-serisic']]

        # add the row to the end of the dataframe
        param_df.loc[param_df.shape[0]] = df_row

    # save to a csv file without an index column
    param_df.to_csv(f'{path}/dataset/param_df.csv', index=False)


if __name__ == '__main__':
    main()
