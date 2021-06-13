import numpy as np

flags = {
    'force': False,
}

gen_params = {
    'image-size': 256,
    'images-number': 800,
    'cmap': 'gray',
    'image-format': 'jpg',
    'valid-split': 0.2,
    'labels': ('lens-redshift'),

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


help_descriptions = {
    's': ('int', 'size of the generated images (square images)', '256'),
    'n': ('int', 'number of images to generate', '200'),
    'c': ('str', 'name of the colormap to use in the images', 'gray'),
    'f': ('str', 'format/extension of the generated images', 'jpg'),
    'v': ('float', 'the ratio of images used for validation of the CNN', '0.2'),

    'lrs': ('float,float', 'range of the red shift smapling for the lens galaxy', '0.5,1.5'),
    'lc': ('float', 'variation of the lens galaxy\'s center', '0.25'),
    'lell': ('float', 'variation in the elliptical components of the lens galaxy', '0.5'),
    'li': ('float,float', 'range of intensity sampling for the lens galaxy\'s energy', '0.5,2.0'),
    'leff': ('float', 'range of sampling of the radius of effect of lens galaxy', '0.5,5.0'),
    'lser': ('float,float', 'range of sampling for Serisic index of the lens galaxy', '1.0,4.0'),
    'lml': ('float,float', 'range for sampling of light to mass ratio of the lens galaxy', '0.1,0.8'),

    'srs': ('float,float', 'range of the red shift smapling for the source galaxy', '0.5,1.5'),
    'sc': ('float', 'variation of the source galaxy\'s center', '0.25'),
    'sell': ('float', 'variation in the elliptical components of the source galaxy', '0.5'),
    'si': ('float,float', 'range of intensity sampling for the source galaxy\'s energy', '0.5,2.0'),
    'seff': ('float', 'range of sampling of the radius of effect of source galaxy', '0.5,5.0'),
    'sser': ('float,float', 'range of sampling for Serisic index of the source galaxy', '1.0,4.0'),
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

    sample['lens-redshift'] = gen_params['lens-redshift-range'][0] + np.random.rand() * np.subtract(np.flip(gen_params['lens-redshift-range']))
    sample['lens-center'] = (np.random.rand()*2 - 1) * gen_params['lens-center-var'], (np.random.rand()*2 - 1) * gen_params['lens-center-var']
    sample['lens-ellip'] = np.random.rand() * gen_params['lens-ellip-var'], np.random.rand() * gen_params['lens-ellip-var']
    sample['lens-intensity'] = gen_params['lens-intensity-range'][0] + np.random.rand() * np.subtract(np.flip(gen_params['lens-intensity-range']))
    sample['lens-effect'] = gen_params['lens-effect-range'][0] + np.random.rand() * np.subtract(np.flip(gen_params['lens-effect-range']))
    sample['lens-serisic'] = np.random.choice(range(gen_params['lens-serisic-range'][0], gen_params['lens-serisic-range'][1]))
    sample['masslight-ratio'] = gen_params['masslight-ratio-range'][0] + np.random.rand() * np.subtract(np.flip(gen_params['masslight-ratio-range']))

    sample['src-redshift'] = gen_params['src-redshift-range'][0] + np.random.rand() * np.subtract(np.flip(gen_params['src-redshift-range']))
    sample['src-center'] = (np.random.rand()*2 - 1) * gen_params['src-center-var'], (np.random.rand()*2 - 1) * gen_params['src-center-var']
    sample['src-ellip'] = np.random.rand() * gen_params['src-ellip-var'], np.random.rand() * gen_params['src-ellip-var']
    sample['src-intensity'] = gen_params['src-intensity-range'][0] + np.random.rand() * np.subtract(np.flip(gen_params['src-intensity-range']))
    sample['src-effect'] = gen_params['src-effect-range'][0] + np.random.rand() * np.subtract(np.flip(gen_params['src-effect-range']))
    sample['src-serisic'] = np.random.choice(range(gen_params['src-serisic-range'][0], gen_params['src-serisic-range'][1]))

    for label in gen_params['labels']:
        sample[label]
