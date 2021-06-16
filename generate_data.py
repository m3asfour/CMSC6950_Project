import os
import sys
import pickle
import autolens as al
import params_and_cli as params
from tqdm import tqdm



def main():
    params.parse_wrapper(sys.argv[1:], 'data')
    check_dataset_dir()

    grid = al.Grid2D.uniform(shape_native=(params.params['data']['grid-size'], params.params['data']['grid-size']), pixel_scales=0.05)
    generate_data_files(grid)


def check_dataset_dir():
    if 'dataset' in os.listdir():
        if params.flags['force']:
            os.system('rm -rf ./dataset/*')
        else:
            sys.tracebacklimit = 0
            print('\n')
            raise Warning('Dataset already exists. Please use "--force" to overwrite it! Exitting...')
    else:
        os.makedirs('dataset')


def generate_data_files(grid):
    params.print_script_description('data')
    for i in tqdm(range(params.params['data']['generated-number']), desc='Generating .pickle files to ./dataset'):
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
        tracer_result = tracer.image_2d_from_grid(grid)
        with open(f'./dataset/tracer_result{i+1}.pickle', 'wb') as f:
            pickle.dump(tracer_result, f)

if __name__ == '__main__':
    main() 