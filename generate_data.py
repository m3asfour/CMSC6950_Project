import autolens as al

import matplotlib.pyplot as plt
import os

grid = al.Grid2D.uniform(
    shape_native=(256, 256),
    pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

lens_galaxy_0 = al.Galaxy(
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

lens_galaxy_1 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.EllExponential(
        centre=(0.00, 0.00),
        elliptical_comps=(0.05, 0.5),
        intensity=1.2,
        effective_radius=0.1,
    ),
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.6), elliptical_comps=(0.05, 0.05), einstein_radius=0.6
    ),
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

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy_0, lens_galaxy_1, source_galaxy])
tracer_img = tracer.image_2d_from_grid(grid).native

fig, ax = plt.subplots(1, 1)
ax.imshow(tracer_img, cmap='magma')
ax.axis('off')


if 'dataset' in os.listdir():
    fig.savefig('./dataset/test.jpg', bbox_inches='tight', pad_inches=0)