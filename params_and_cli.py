"""
    This script contains parameters of the package and the text to print to the cli
    This script doesn't have and main routine and won't do anything if run 
"""

# import modules
import sys
import os
import numpy as np


# flags to perform some conditional code blocks in other scripts
flags = {
    'force': False,
    'model-figures': True,
}


# the cli hyper-parameters that control the data generation, model training, and figures plotting
# "data" key is for the script "generate_data.py"
# "model-figures" key is for the script "model_and_figures.py"
params = {
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

    'model-figures': {
            'valid-split': 0.2,
            'test-split': 0.1,
            'labels': ['src-redshift', 'src-center-x', 'src-center-y'],
            'epochs': 50,
            'learning-rate': 0.00001,
            'batch-size': 20,

            'figure-format': 'jpg',
            'cmap': 'magma',
            'subplots-number': 6,
    }
}


# the cli shortcuts for hyper-parameters
# "data" key is for the script "generate_data.py"
# "model-figures" key is for the script "model_and_figures.py"
shortcuts = {
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

    'model-figures': {
            'v': 'valid-split',
            't': 'test-split',
            'l': 'labels',
            'e': 'epochs',
            'mlr': 'learning-rate',
            'b': 'batch-size',

            'f': 'figure-format',
            'c': 'cmap',
            'sn': 'subplots-number'
    },    
}


# the types for the passed cli arguments for hyper-parameters
# "data" key is for the script "generate_data.py"
# "model-figures" key is for the script "model_and_figures.py"
params_types = {
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
    'test-split': float,
    'labels': list,
    'epochs': int,
    'learning-rate': float,
    'batch-size': int,

    'figure-format': str,
    'cmap': str,
    'subplots-number': int,
}


# the "--help" descriptions of cli arguments for hyper-parameters
# "data" key is for the script "generate_data.py"
# "model-figures" key is for the script "model_and_figures.py"
help_descriptions = {
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

    'v': ('float', 'the ratio of images used for validation of the CNN', str(params['model-figures']['valid-split'])),
    't': ('float', 'the ratio of images used for testing of the CNN', str(params['model-figures']['test-split'])),
    'l': ('str,...,str', 'the labels to train the neural network with', str(params['model-figures']['labels'])),
    'e': ('int', 'the epochs for the CNN model training', str(params['model-figures']['epochs'])),
    'mlr': ('float', 'the learning rate for the CNN model training', str(params['model-figures']['learning-rate'])),
    'b': ('int', 'the batch size for the CNN model training', str(params['model-figures']['batch-size'])),

    'f': ('str', 'format/extension of the generated images', str(params['model-figures']['figure-format'])),
    'c': ('str', 'name of the colormap to use in the images', str(params['model-figures']['cmap'])),
    'sn': ('int', 'number of subplots to use in the generated figures', str(params['model-figures']['subplots-number'])),
}


# the cli hints for the scripts (before, and after the help table or after running the script)
# "data" key is for the script "generate_data.py"
# "model-figures" key is for the script "model_and_figures.py"
# headers -> printed before the help table, footers -> printed after the help table
# defaults -> printed after the script is run
cli_hints = {
    'headers': {
        'data': '\n\nThis script is used to generate gravitational lens files in ./dataset directory.\nYou can control the generation process using the following optional arguments',
        'model-figures': '\n\nThis script is used to create a CNN model, train it, and generate figures to ./figures.\nYou can control the generation process using the following optional arguments',
    },

    'footers': {
        'data': '\nuse "--argument=value" or "-shortcut=value" to specify the values. For example "python generate_data.py --src-intensity-range=0.3,1.2"\n',
        'model-figures': '\nuse "--argument=value" or "-shortcut=value" to specify the values. For example "python model_and_figures.py -epochs=100 -f=png"\n',
    },

    'defaults': {
        'data': '\n\nThis script is used to generate gravitational lens files in ./dataset directory\nyou can use "python generate_data.py --help" to view all the optional arguments to control the data generation.\n',
        'model-figures': '\n\nThis script is used to create a CNN model, train it, and generate figures in ./figures directory\nyou can use "python model_and_figures.py --help" to view all the optional arguments.\n'
    }
}


# the descriptions of possible labels to use
labels_description = {
    'src-redshift': 'The red shift level of the source galaxy',
    'src-center-x': 'The x position of the source galaxy in the image',
    'src-center-y': 'The y position of the source galaxy in the image',
    'src-ellip-x': 'The x component of the source galaxy shape',
    'src-ellip-y': 'The y component of the source galaxy shape',
    'src-intensity': 'The intensity of the source galaxy\'s energy/light',
    'src-effect':  'The radius of effect of the source galaxy\'s energy/light',
    'src-serisic': 'The Serisic index of the source galaxy'
}



def generate_sample():
    """generates a sample/configuration of the gravitational lens problem

    Returns:
        dict: a dictionary containing the configuration values of the sample
    """
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
    """prints the allowed labels description to the cli
    """
    print('\n\n')
    w = os.get_terminal_size()[0]   # get the terminal width to center the text
    print(f'{"These are the allowed parameters to use":{w}}\n')
    print(f'\n{"Label":<25} {"Description":<65}')
    print(f'{"":-<25} {"":-<65}')
    for label, desc in labels_description.items():
        print(f'{f"{label}":<25} {desc:<65}')
    exit()


def print_script_description(script):
    """prints the description text of the passed script to to the cli

    Args:
        script (str): the script name/key
    """

    w = os.get_terminal_size()[0]   # get the terminal width to center the text
    # center the multiple lines of the description
    for line in cli_hints['defaults'][script].split('\n'):
        print(f'{line:^{w}}')


def print_help(script):
    """prints the help table of the script to the cli

    Args:
        script (str): the script name/key.
    """
    w = os.get_terminal_size()[0]   # get terminal width to center the header and footer
    for line in cli_hints['headers'][script].split('\n'):
        print(f'{line:^{w}}')

    # print the help table
    print(f'\n{"Argument":<25} {"Shortcut":<10} {"Type":<15} {"Description":<65} {"Default Value":<15}')
    print(f'{"":-<25} {"":-<10} {"":-<15} {"":-<65} {"":-<15}')
    for shortcut in shortcuts[script]:
        arg_type, arg_desc, arg_default = help_descriptions[shortcut]
        if arg_default[0] != '[':   # if default value isn't a list of multiple values
            print(f'{f"--{shortcuts[script][shortcut]}":<25} {f"-{shortcut}":<10} {arg_type:<15} {arg_desc:<65} {arg_default:<15}')
        else:
            for idx, val in enumerate(arg_default.split(',')):  # loop over values in default value list
                if idx == 0:    # print the other fields for in the same line as first value
                    print(f'{f"--{shortcuts[script][shortcut]}":<25} {f"-{shortcut}":<10} {arg_type:<15} {arg_desc:<65} {val:<15}')
                else:           # print remaining values without the other fields
                    print(f'{"":<25} {"":<10} {"":<15} {"":<65} {val:<15}')

    for line in cli_hints['footers'][script].split('\n'):
        print(f'{line:^{w}}')
    exit()


def parse_wrapper(passed_args, script):
    """parse the arguments and catch and thrown errors before displaying them

    Args:
        passed_args (list): the list of the passed arguments to the cli
        script (str): the name of the script calling the parse_wrapper function

    Raises:
        error: error occured while parsing the arguments
    """
    if passed_args:
        error = parse_args(passed_args, script=script)
        if error:
            sys.tracebacklimit = 0  # supress printing traceback before raising error
            print('\n')
            raise error


def parse_args(args, script):
    """parses the arguments passed to the cli

    Args:
        args (list): list of the passed arguments to the cli
        script (str): the name of the script calling the parse_args function

    Returns:
        Error: the error occurred while parsing the arguments
    """
    for arg in args:
        if not '=' in arg:  # if not a argument with a value (a flag)
            try:
                if arg[2:] == 'labels': # print the allowed labels to use
                    print_labels()
                elif arg[2:] == 'help': # print the help table
                    print_help(script)

                flags[arg[2:]] = not flags[arg[2:]] # toggle the flag value
            except KeyError as error:
                return KeyError((f'Flag not recognized "{arg[2:]}"!'))
        else:   # if an argument with a value
            key, val = arg.split('=')   # split the argument name from the value
            try:
                if '--' == key[:2]: # remove "--" from argument
                    key = key[2:]
                elif '-' == key[0]: # remove "-" from shortcut
                    key = shortcuts[script][key[1:]]
                else:
                    return InputError('Only parameters preceeded by "-" or "--" are allowed. Please use "python generate_data.py --help" for more info!')

                val_type = params_types[key]    # the proper type for the argument value
                if key != 'labels': # if the argument isn't setting the labels
                    val_cast = val_type(val) if val_type != tuple else val_type([float(v) for v in val.split(',')])
                else:
                    val_cast = val_type([v for v in val.split(',')])
                params[script][key] = val_cast  # set the casted value to the parameters dictionary

            except KeyError:
                return KeyError(f'Passed parameter not recognized: "{key}"!')

            except ValueError as error:
                return ValueError(f'Invalid value passed for the parameter "{key}"\n{error}!')


# a custom error type if the user passed a bad argument
class InputError(Exception):
    pass
