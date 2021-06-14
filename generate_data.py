import os
import sys
import shutil

import autolens as al
import generation_parameters as params
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm



class InputError(Exception):
    pass


def parse_args(args):
    for arg in args:
        if not '=' in arg:
            try:
                if arg[2:] == 'labels':
                    params.print_labels()
                elif arg[2:] == 'help':
                    params.print_help()

                params.flags[arg[2:]] = not params.flags[arg[2:]]
            except KeyError as error:
                return KeyError((f'Flag not recognized "{arg[2:]}"!'))
        else:
            key, val = arg.split('=')
            try:
                if '--' == key[:2]:
                    key = key[2:]
                elif '-' == key[0]:
                    key = params.args_shortcuts[key[1:]]
                else:
                    return InputError('Only parameters preceeded by "-" or "--" are allowed. Please use "python generate_data.py --help" for more info!')

                val_type = params.parameters_types[key]
                if key != 'labels':
                    val_cast = val_type(val) if val_type != tuple else val_type([float(v) for v in val.split(',')])
                else:
                    val_cast = val_type([v for v in val.split(',')])
                params.gen_params[key] = val_cast

            except KeyError:
                return KeyError(f'Passed parameter not recognized: "{key}"!')

            except ValueError as error:
                return ValueError(f'Invalid value passed for the parameter "{key}"\n{error}!')


def main():
    passed_args = sys.argv[1:]
    if passed_args:
        error = parse_args(passed_args)
        if error:
            sys.tracebacklimit = 0
            print('\n')
            raise error

    check_dataset_dir()

    grid = al.Grid2D.uniform(shape_native=(params.gen_params['image-size'], params.gen_params['image-size']), pixel_scales=0.05)
    info_df = pd.DataFrame(data=[], columns=['img_path', 'subset', *params.generate_sample().keys()])
    generate_images(grid, info_df)


def check_dataset_dir():
    if 'dataset' in os.listdir():
        if len(os.listdir('dataset/')) == 0 or params.flags['force']:
            shutil.rmtree('dataset/')
            os.makedirs('dataset')
        else:
            sys.tracebacklimit = 0
            print('\n')
            raise Warning('Dataset already exists. Please use "--force" to overwrite it! Exitting...')
    else:
        os.makedirs('dataset')


def generate_images(grid, info_df):
    fig, ax = plt.subplots(1, 1, figsize=(params.gen_params['image-size'], params.gen_params['image-size']), dpi=1)
    ax.set_position([0, 0, 1, 1])

    for i in tqdm(range(params.gen_params['images-number']), desc='Generating images to ./dataset'):
        sample = params.generate_sample()

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
            dark=al.mp.SphNFW(centre=(0.0, 0.0), kappa_s=0.08, scale_radius=30.0),
        )

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

        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
        tracer_img = tracer.image_2d_from_grid(grid).native
        
        ax.imshow(tracer_img, cmap=params.gen_params['cmap'])
        ax.axis('off')
        
        fig.savefig(f'./dataset/img{i+1}.jpg')
    plt.close(fig)


if __name__ == '__main__':
    main() 