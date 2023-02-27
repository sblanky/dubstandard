Attempts to determine optimum Dubinin (Astakhov or Radushkevich) volume from an isotherm. Work in progress.

# How it works

1. The Dubinin transform is performed on the isotherm. Exponent can be specified or automatically optimised. 
2. Transformed isotherm sliced into pieces. Dubinin pore volume and related parameters calculated from each slice.
3. The dictionary of results is filtered; automatic filter parameters;

| Parameter  | Value         | Explanation	|
| ---------------- | ---------------- | ------------ |
| 'bounds'   | [1, 3]                | bounds of exponent	|
| 'curvature_limit'  | 1             | amount of curvature in selected region 	|
| 'p_limits'         | [0, 0.1]              | pressure range of isotherm to select 	|
| 'max_capacity'     | (total pore capacity)         | maximm isotherm loading in liquid volume |
| 'min_points'      | 10            | minimum number of points in selected region 	|	
| 'corr_coef'        | 0.999         | minimum correlation coefficient of linear regression |


4. The fitting range with the lowest pore volume is selected as the optimum.
5. Results are exported.
# Installation

- Install [pygaps](https://github.com/pauliacomi/pyGAPS/)
- Clone this rep
- `cd` into the cloned repo, and do `pip install .`

# Basic use

Run the function `analyseDR` on a pygaps isotherm, e.g.

```py

import pygaps.parsing as pgp
from dubstandard.dubinin import analyseDR

file = '/path/to/file.aif'
isotherm = pgp.isotherm_from_aif(
		isotherm,
		**{} # filter parameters can be changed here.
		)
analyseDR(isotherm)

```

Currently the easiest way to run the program is to start with an isotherm in `.aif` format. Example isotherms (copied from [betsi](https://github.com/nakulrampal/betsi-gui)) can be found in [`./example/aif/`](./example/aif/). Results from both Dubinin-Radushkevich and optimised Dubinin-Astakhov treatment can also be found in [`./example/`](./example/). 


# Testing on example isotherm set

Only a few isotherms successfully generated a Dubinin pore volume. It is possible that results could be attained by increasing bounds of exponent.
