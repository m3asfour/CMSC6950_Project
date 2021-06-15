import sys
import numpy as np

flags = {
    'force': False,
}


params = {
    'figures': {
            'figure-format': 'jpg',
            'cmap': 'magma',
    },

    'data': {
            'grid-size': 256,
            'generated-number': 100,

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
    },

    'training': {
            'valid-split': 0.2,
            'labels': ('lens-redshift'),
    }
}

shortcuts = {
    'figures': {
            'f': 'figure-format',
            'c': 'cmap',
    },

    'data': {
            's': 'grid-size',
            'n': 'generated-number',

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
    },

    'training': {
            'v': 'valid-split',
            'l': 'labels',
    },    
}

params_types = {
    'figure-format': str,
    'cmap': str,

    'grid-size': int,
    'generated-number': int,

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
    
    'valid-split': float,
    'labels': tuple,
}


help_descriptions = {
    'f': ('str', 'format/extension of the generated images', str(params['figures']['figure-format'])),
    'c': ('str', 'name of the colormap to use in the images', str(params['figures']['cmap'])),
    
    's': ('int', 'size of the generated images (square images)', str(params['data']['grid-size'])),
    'n': ('int', 'number of images to generate', str(params['data']['generated-number'])),

    'lrs': ('float,float', 'range of the red shift smapling for the lens galaxy', str(params['data']['lens-redshift-range'])),
    'lc': ('float', 'variation of the lens galaxy\'s center', str(params['data']['lens-center-var'])),
    'lell': ('float', 'variation in the elliptical components of the lens galaxy', str(params['data']['lens-ellip-var'])),
    'li': ('float,float', 'range of intensity sampling for the lens galaxy\'s energy', str(params['data']['lens-intensity-range'])),
    'leff': ('float', 'range of sampling of the radius of effect of lens galaxy', str(params['data']['lens-effect-range'])),
    'lser': ('float,float', 'range of sampling for Serisic index of the lens galaxy', str(params['data']['lens-serisic-range'])),
    'lml': ('float,float', 'range for sampling of light to mass ratio of the lens galaxy', str(params['data']['masslight-ratio-range'])),

    'srs': ('float,float', 'range of the red shift smapling for the source galaxy', str(params['data']['src-redshift-range'])),
    'sc': ('float', 'variation of the source galaxy\'s center', str(params['data']['src-center-var'])),
    'sell': ('float', 'variation in the elliptical components of the source galaxy', str(params['data']['src-ellip-var'])),
    'si': ('float,float', 'range of intensity sampling for the source galaxy\'s energy', str(params['data']['src-intensity-range'])),
    'seff': ('float', 'range of sampling of the radius of effect of source galaxy', str(params['data']['src-effect-range'])),
    'sser': ('float,float', 'range of sampling for Serisic index of the source galaxy', str(params['data']['src-serisic-range'])),

    'v': ('float', 'the ratio of images used for validation of the CNN', str(params['training']['valid-split'])),
    'l': ('str,...,str', 'the labels to train the neural network with', str(params['training']['labels'])),
}


labels_description = {
    'src-redshift': 'The red shift level of the source galaxy',
    'src-center': 'The (x,y) position of the source galaxy in the image',
    'src-ellip': 'The (x component, y component) of the source galaxy shape',
    'src-intensity': 'The intensity of the source galaxy\'s energy/light',
    'src-effect':  'The radius of effect of the source galaxy\'s energy/light',
    'src-serisic': 'The Serisic index of the source galaxy'
}


def generate_sample():
    sample = {}

    sample['lens-redshift'] = params['data']['lens-redshift-range'][0] + np.random.rand() * np.subtract(*np.flip(params['data']['lens-redshift-range']))
    sample['lens-center'] = (np.random.rand()*2 - 1) * params['data']['lens-center-var'], (np.random.rand()*2 - 1) * params['data']['lens-center-var']
    sample['lens-ellip'] = np.random.rand() * params['data']['lens-ellip-var'], np.random.rand() * params['data']['lens-ellip-var']
    sample['lens-intensity'] = params['data']['lens-intensity-range'][0] + np.random.rand() * np.subtract(*np.flip(params['data']['lens-intensity-range']))
    sample['lens-effect'] = params['data']['lens-effect-range'][0] + np.random.rand() * np.subtract(*np.flip(params['data']['lens-effect-range']))
    sample['lens-serisic'] = np.random.choice(range(params['data']['lens-serisic-range'][0], params['data']['lens-serisic-range'][1]))
    sample['masslight-ratio'] = params['data']['masslight-ratio-range'][0] + np.random.rand() * np.subtract(*np.flip(params['data']['masslight-ratio-range']))

    sample['src-redshift'] = params['data']['src-redshift-range'][0] + np.random.rand() * np.subtract(*np.flip(params['data']['src-redshift-range']))
    sample['src-center'] = (np.random.rand()*2 - 1) * params['data']['src-center-var'], (np.random.rand()*2 - 1) * params['data']['src-center-var']
    sample['src-ellip'] = np.random.rand() * params['data']['src-ellip-var'], np.random.rand() * params['data']['src-ellip-var']
    sample['src-intensity'] = params['data']['src-intensity-range'][0] + np.random.rand() * np.subtract(*np.flip(params['data']['src-intensity-range']))
    sample['src-effect'] = params['data']['src-effect-range'][0] + np.random.rand() * np.subtract(*np.flip(params['data']['src-effect-range']))
    sample['src-serisic'] = np.random.choice(range(params['data']['src-serisic-range'][0], params['data']['src-serisic-range'][1]))

    return sample

def print_labels():
    exit()


# print('\n\nThis script is used to generate gravitational lens images in ./dataset directory.\nYou can control the generation process using the following optional arguments')


def print_help(script='data'):
    print(f'\n{"Argument":<25} {"Shortcut":<10} {"Type":<15} {"Description":<65} {"Default Value":<15}')
    print(f'{"":-<25} {"":-<10} {"":-<15} {"":-<65} {"":-<15}')
    for shortcut in shortcuts[script]:
        arg_type, arg_desc, arg_default = help_descriptions[shortcut]
        print(f'{f"--{shortcuts[script][shortcut]}":<25} {f"-{shortcut}":<10} {arg_type:<15} {arg_desc:<65} {arg_default:<15}')
    print('\nuse "--argument=value" or "-shortcut=value" to specify the values. For example "python generate_data.py --src-intensity-range=0.3,1.2 -f=png"\n')
    exit()


class InputError(Exception):
    pass


def parse_wrapper(passed_args, script):
    if passed_args:
        error = parse_args(passed_args, script=script)
        if error:
            sys.tracebacklimit = 0
            print('\n')
            raise error

def parse_args(args, script):
    for arg in args:
        if not '=' in arg:
            try:
                if arg[2:] == 'labels':
                    print_labels()
                elif arg[2:] == 'help':
                    print_help(script)

                flags[arg[2:]] = not flags[arg[2:]]
            except KeyError as error:
                return KeyError((f'Flag not recognized "{arg[2:]}"!'))
        else:
            key, val = arg.split('=')
            try:
                if '--' == key[:2]:
                    key = key[2:]
                elif '-' == key[0]:
                    key = shortcuts[script][key[1:]]
                else:
                    return InputError('Only parameters preceeded by "-" or "--" are allowed. Please use "python generate_data.py --help" for more info!')

                val_type = params_types[key]
                if key != 'labels':
                    val_cast = val_type(val) if val_type != tuple else val_type([float(v) for v in val.split(',')])
                else:
                    val_cast = val_type([v for v in val.split(',')])
                params[script][key] = val_cast

            except KeyError:
                return KeyError(f'Passed parameter not recognized: "{key}"!')

            except ValueError as error:
                return ValueError(f'Invalid value passed for the parameter "{key}"\n{error}!')
