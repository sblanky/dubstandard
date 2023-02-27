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
                optimum_criteria='max_corr_coef',
                output_dir=output_dir,
                exp=exp,
                verbose=True,
            )
        except ValueError as e:
            print(isotherm.material)
            print(e)
            continue
