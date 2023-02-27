import glob
from dubstandard.dubinin import analyseDR
import pygaps.parsing as pgp

path = './aif/'

for f in glob.glob(f'{path}*.aif'):
    isotherm = pgp.isotherm_from_aif(f)
    for exp in [None, 2]:
        if exp is None:
            output_dir = './DAoptimum/'
        else:
            output_dir = './DR/'
        try:
            analyseDR(
                isotherm,
                output_dir=output_dir,
                verbose=True,
                exp=exp,
                **{
                    'min_corr_coef': 0.9,
                    'curvature': 10,
                    'min_points': 5,
                    'bounds': [1, 3],
                }
            )
        except ValueError as e:
            print(isotherm.material)
            print(e)
            continue
