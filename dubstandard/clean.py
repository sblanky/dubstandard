import numpy as np

import pygaps.parsing as pgp
from pygaps import PointIsotherm


def strictly_increasing(List):
    """
    Takes in a list, and deletes the values that are not strictly increasing.
    Then returns modified list.
    """
    increasing = [x < y for x, y in zip(List, List[1:])]
    increasing.append(increasing[-1])
    return increasing


def increasing_pressure(
    isotherm: PointIsotherm
):
    """
    Takes in PointIsotherm, and removes decreasing pressures.
    Pressure and loading are returned.
    """
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
    """
    Applies cleaning function to isotherm.
    Currently only ensures increasing.
    """
    loading, pressure = increasing_pressure(isotherm)
    return loading, pressure


if __name__ == "__main__":
    inPath = '../aif/'
    file = 'Al_fumarate.aif'
    isotherm = pgp.isotherm_from_aif(f'{inPath}{file}')
    isotherm.convert()
    print(file)
    print(isotherm)
    print(clean_isotherm(isotherm))
