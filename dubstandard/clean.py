import numpy as np

import pygaps.parsing as pgp
from pygaps import PointIsotherm


def strictly_increasing(List):
    increasing = [x < y for x, y in zip(List, List[1:])]
    increasing.append(increasing[-1])
    return increasing


def increasing_pressure(isotherm):
    increasing = strictly_increasing(isotherm.pressure())
    increasing_index = [
        i for i, x in enumerate(increasing) if x
    ]
    pressure = [
        x for i, x in enumerate(isotherm.pressure())
        if i in increasing_index
    ]
    loading = isotherm.loading_at(pressure)

    return np.array(loading), np.array(pressure)


def clean_isotherm(isotherm):
    loading, pressure = increasing_pressure(isotherm)
    return loading, pressure


if __name__ == "__main__":
    inPath = '/home/pcxtsbl/CodeProjects/labcore_upload/robert/revise/aif/'
    file = 'LAC2800.aif'
    isotherm = pgp.isotherm_from_aif(f'{inPath}{file}')
    isotherm.convert()
    print(file)
    print(isotherm)
    print(clean_isotherm(isotherm))
