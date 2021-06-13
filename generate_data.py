import os
import sys
import shutil
from types import TracebackType

import autolens as al
import matplotlib.pyplot as plt


class InputError(Exception):
    pass

flags = {
    'force': False,
}

generation_parameters = {
    'image-size': 256,
    'images-number': 800,
    'cmap': 'gray',
    'image-format': 'jpg',
    'valid-split': 0.2,

    'lens-redshift-range': (0.5, 1.5),
    'lens-center-var': 0.25,
    'lens-ellip-var': 0.5,
    'lens-intensity-range': (0.5, 2.0),
    'lens-effect-range': (0.5, 5.0),
    'lens-serisic-range': (1, 4),
    'masslight-ratio-range': (0.1, 0.8),

    'src-redshift-range': (0.5, 1.5),
    'src-center-var': 0.25,
    'src-ellip-var': 0.5,
    'src-intensity-range': (0.5, 2.0),
    'src-effect-range': (0.5, 5.0),
    'src-serisic-range': (1, 4),
}

args_shortcuts = {
    's': 'image-size',
    'n': 'images-number',
    'c': 'cmap',
    'f': 'image-format',
    'v': 'valid-split',

    'lrs': 'lens-redshift-range',
    'lc': 'lens-center-var',
    'lell': 'lens-ellip-var',
    'li': 'lens-intensity-range',
    'leff': 'lens-effect-range',
    'lser': 'lens-serisic-range',
    'lml': 'masslight-ratio-range',

    'srs': 'src-redshift-range',
    'sc': 'src-center-var',
    'sell': 'src-ellip-var',
    'si': 'src-intensity-range',
    'seff': 'src-effect-range',
    'sser': 'src-serisic-range',
}

parameters_types = {
    'image-size': int,
    'images-number': int,
    'cmap': str,
    'image-format': str,
    'valid-split': float,

    'lens-redshift-range': tuple,
    'lens-center-var': float,
    'lens-ellip-var': float,
    'lens-intensity-range': tuple,
    'lens-effect-range': tuple,
    'lens-serisic-range': tuple,
    'masslight-ratio-range': tuple,

    'src-redshift-range': tuple,
    'src-center-var': float,
    'src-ellip-var': float,
    'src-intensity-range': tuple,
    'src-effect-range': tuple,
    'src-serisic-range': tuple,
}

def parse_args(args):
    for arg in args:
        if not '=' in arg:
            try:
                flags[arg[2:]] = not flags[arg[2:]]
            except KeyError as error:
                return KeyError((f'Flag not recognized "{arg[2:]}"!'))
        else:
            key, val = arg.split('=')
            try:
                if '--' == key[:2]:
                    key = key[2:]
                elif '-' == key[0]:
                    key = args_shortcuts[key[1:]]
                else:
                    return InputError('Only parameters preceeded by "-" or "--" are allowed. Please use "python generate_data.py --help" for more info!')

                val_type = parameters_types[key]
                val_cast = val_type(val) if val_type != tuple else val_type([float(v) for v in val.split(',')])
                generation_parameters[key] = val_cast
            except KeyError:
                return KeyError(f'Passed parameter not recognized: "{key}"!')

            except ValueError as error:
                return ValueError(f'Invalid value passed for the parameter "{key}"\n{error}!')


passed_args = sys.argv[1:]
if passed_args:
    error = parse_args(passed_args)
    if error:
        sys.tracebacklimit = 0
        print('\n')
        raise error


if 'dataset' in os.listdir():
    if len(os.listdir('dataset/')) == 0 or flags['force']:
        shutil.rmtree('dataset/')
        os.makedirs('dataset')
    else:
        sys.tracebacklimit = 0
        print('\n')
        raise Warning('Dataset already exists. Please use "--force" to overwrite it! Exitting...')
else:
    os.makedirs('dataset')


grid = al.Grid2D.uniform(shape_native=(256, 256), pixel_scales=0.05)


lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lmp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.05),
        intensity=0.5,
        effective_radius=0.3,
        sersic_index=3.5,
        mass_to_light_ratio=0.6,
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
    redshift=2.0,
    bulge=al.lp.EllSersic(
        centre=(0.0, 50),
        elliptical_comps=(0.0, 5),
        intensity=2,
        effective_radius=0.6,
        sersic_index=1,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
tracer_img = tracer.image_2d_from_grid(grid).native

fig, ax = plt.subplots(1, 1)
ax.imshow(tracer_img, cmap='magma')
ax.axis('off')

print('done')
fig.savefig('./dataset/test.jpg', bbox_inches='tight', pad_inches=0)