"""
    dubstandard; applying the BETSI approach to calculate Dubinin micropore
    volumes from isotherms.
    Copyright (C) 2023 L. Scott Blankenship.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from scipy.interpolate import PchipInterpolator
from scipy.signal import argrelextrema
from scipy import optimize, stats, constants
import numpy as np
import matplotlib.pyplot as plt

import logging
import warnings
from pathlib import Path
from pprint import pprint

import pygaps.parsing as pgp
import pygaps.characterisation as pgc
from pygaps.graphing.calc_graphs import dra_plot
from pygaps.utilities.exceptions import CalculationError

from clean import clean_isotherm


def log_p_exp(
    pressure,
    exp
):
    r"""
    Converts pressure array to log of array, raised to exponent
    """
    return (-np.log(pressure))**exp


class DubininResult:
    """
    Store all possible Dubinin results from an isotherm,
    as well as curvature, and other information for filtering
    later.
    """
    def __init__(
        self,
        isotherm,
        clean=False,
        exp: float = None,
        **kwargs,
    ):
        def zero_matrix(dim):
            return np.zeros([dim, dim])

        self.material = isotherm.material
        self.adsorbate = isotherm.adsorbate
        self.iso_temp = isotherm.temperature
        self.molar_mass = self.adsorbate.molar_mass()
        self.liquid_density = self.adsorbate.liquid_density(self.iso_temp)

        isotherm.convert(
            pressure_mode='relative',
            loading_unit='mol',
            loading_basis='molar',
            material_unit='g',
            material_basis='mass',
        )
        self.loading, self.pressure = clean_isotherm(isotherm)

        plateau_pressure = kwargs.get('plateau_pressure', 0.95)
        if max(isotherm.pressure()) < plateau_pressure:
            plateau_pressure = max(isotherm.pressure())

        if 'max_loading' in kwargs:
            self.plateau_loading = kwargs['max_loading']
        else:
            self.plateau_loading = isotherm.loading_at(
                pressure=plateau_pressure,
                branch='ads',
                pressure_mode='relative',
                loading_unit='mol',
                loading_basis='molar',
                material_unit='g',
                material_basis='mass',
            )
        self.characteristic_loading = self.plateau_loading / np.e
        self.characteristic_pressure = isotherm.pressure_at(
            loading=self.characteristic_loading,
            branch='ads',
            pressure_mode='relative',
        )
        self.characteristic_energy = (
            constants.gas_constant * self.iso_temp *
            np.log(1 / self.characteristic_pressure)
        ) / 1000
        self.total_pore_volume = (
            self.plateau_loading * self.molar_mass /
            self.liquid_density
        )

        self.log_v = pgc.dr_da_plots.log_v_adj(
            self.loading,
            self.molar_mass,
            self.liquid_density,
        )

        num_points = len(self.pressure)

        self.point_count = zero_matrix(num_points)
        self.fit_grad = zero_matrix(num_points)
        self.fit_intercept = zero_matrix(num_points)
        self.fit_rsquared = zero_matrix(num_points)
        self.p_val = zero_matrix(num_points)
        self.stderr = zero_matrix(num_points)
        self.second_derivative = zero_matrix(num_points)

        self.potentials = zero_matrix(num_points)
        self.log_sq_potential = zero_matrix(num_points)
        self.pore_volume = zero_matrix(num_points)

        self.pressure_max = zero_matrix(num_points)
        self.pressure_min = zero_matrix(num_points)

        self.filter_volume = zero_matrix(num_points)
        self.filter_pressure = zero_matrix(num_points)

        self.rouq_y = self.loading * (1. - self.pressure)
        self.rouq_knee_idx = np.argmax(np.diff(self.rouq_y) < 0)

        self.ultrarouq_y = (
            np.log(self.loading) /
            np.log(1 - self.pressure)
        )
        ultrarouq_knee_idx = argrelextrema(
            self.ultrarouq_y,
            np.less
        )[0]
        if len(ultrarouq_knee_idx) > 1:
            ultrarouq_knee_idx = ultrarouq_knee_idx[-1]
        if (
            ultrarouq_knee_idx.size == 0 or
            ultrarouq_knee_idx >= self.rouq_knee_idx
        ):
            ultrarouq_knee_idx = np.array([0])

        self.ultrarouq_knee_idx = ultrarouq_knee_idx[
            ultrarouq_knee_idx < self.rouq_knee_idx
        ]


        self.rouq_expand = zero_matrix(num_points)

        self.rouq_linear = zero_matrix(num_points)

        def dr_fit(exp, ret=False):
            slope, intercept, corr_coef, _, stderr = stats.linregress(
                log_p_exp(self.pressure, exp), self.log_v
            )
            if ret:
                return slope, intercept, corr_coef
            return stderr

        if exp is None:
            bounds = kwargs.get('bounds', [1, 3])
            res = optimize.minimize_scalar(
                dr_fit,
                bounds=bounds,
                method='bounded'
            )
            if not res.success:
                raise CalculationError(
                    'Could not obtain a linear fit on the data.'
                )
            self.exp = res.x
        else:
            self.exp = exp

        self.log_p_exp = log_p_exp(self.pressure, self.exp)

        self._compute_dubinin_data(
            self.pressure,
            self.loading,
        )

    def _compute_dubinin_data(
        self,
        pressure,
        loading,
    ):

        num_points = len(pressure)
        self.result = {}
        for i in range(num_points):
            for j in range(i+1, num_points):
                self.point_count[i, j] = j - i

                self.pressure_min[i, j] = self.pressure[i]
                self.pressure_max[i, j] = self.pressure[j]

                if (
                    self.rouq_knee_idx > j and
                    self.ultrarouq_knee_idx < i
                ):
                    self.rouq_expand[i, j] = 1

                with warnings.catch_warnings(record=True) as w:
                    (
                        fit_grad,
                        fit_intercept,
                        fit_rvalue,
                        p_val,
                        stderr,
                    ) = stats.linregress(
                        self.log_p_exp[i:j],
                        self.log_v[i:j]
                    )

                    self.fit_grad[i, j] = fit_grad
                    self.fit_intercept[i, j] = fit_intercept
                    self.fit_rsquared[i, j] = fit_rvalue**2
                    self.p_val[i, j] = p_val
                    self.stderr[i, j] = stderr

                    pore_filled_molar = np.exp(fit_intercept)

                    self.pore_volume[i, j] = np.exp(fit_intercept)
                    if not np.isfinite(self.pore_volume[i, j]):
                        self.pore_volume[i, j] = 0

                    self.potentials[i, j] = (
                        (constants.gas_constant * self.iso_temp) /
                        (-fit_grad)**(1 / self.exp) / 1000
                    )
                    if not np.isfinite(self.potentials[i, j]):
                        self.potentials[i, j] = 0
                    self.log_sq_potential[i, j] = 2 * np.log(
                        self.potentials[i, j]
                    )

                    if len(w) > 0:
                        continue


class DubininFilteredResults:
    r"""
    Apply filter to results from DubininResult.
    Filters can be changed using kwargs.
    """

    def __init__(
        self,
        dubinin_result,
        optimum_criteria: str = 'max_points',
        verbose=False,
        **kwargs,
    ):
        self.__dict__.update(dubinin_result.__dict__)
        self.verbose=verbose

        filter_mask = np.ones_like(dubinin_result.point_count)

        min_points = kwargs.get('min_points', 10)
        filter_mask = filter_mask * (dubinin_result.point_count > min_points)

        min_r2 = kwargs.get('min_r2', 0.99)
        filter_mask = filter_mask * (dubinin_result.fit_rsquared > min_r2)

        filter_mask = filter_mask * dubinin_result.rouq_expand

        max_volume = kwargs.get('max_volume', dubinin_result.total_pore_volume)
        filter_mask = filter_mask * (dubinin_result.pore_volume <= max_volume)

        self.filter_params = kwargs

        self.has_valid_volumes = False
        if np.sum(filter_mask) != 0:
            self.has_valid_volumes = True
            self.pore_volume_filtered = self.pore_volume * filter_mask
            self.valid_indices = np.where(
                self.pore_volume_filtered > 0
            )
            self.potential_filtered = self.potentials * filter_mask

            self.num_valid = len(self.valid_indices[0])

            self.valid_potentials = self.potential_filtered[self.valid_indices]
            self.valid_volumes = self.pore_volume_filtered[self.valid_indices]
            self.valid_point_counts = self.point_count[self.valid_indices]
            self.valid_rsquared = self.fit_rsquared[self.valid_indices]

            self._find_nearest_idx()

            self.volume = self.pore_volume_filtered[self.k]
            self.rsquared = self.fit_rsquared[self.k]
            self.ans_potential = self.potentials[self.k]
            self.opt_point_count = self.point_count[self.k]

    def _find_nearest_idx(self):
        closest = np.argmin(
            np.abs(self.valid_potentials - self.characteristic_energy)
        )
        self.k = self.valid_potentials.index(closest)
        self.k = np.unravel_index(
            np.argmin(
                np.abs(self.valid_potentials - self.characteristic_energy),
            ), self.valid_potentials.shape
        )[0]
        self.i = np.unravel_index(
            self.
        )
        print(self.k)
        #self.i = coords[0]
        #self.j = coords[1]


    def export(self, filepath, verbose):
        filepath = Path(filepath)

        with (filepath / 'filter_summary.json').open('w') as fp:
            pprint(self.filter_params, fp)

        with (filepath / 'results.txt').open('w') as fp:
            if self.has_valid_volumes:
                print(
                    f'Optimum Dubinin fit with exponent {self.exp}: ',
                    file=fp
                )
                print(f'Micropore volume: {self.volume}', file=fp)
                print(f'total points: {self.opt_point_count}', file=fp)
                print(f'r-squared: {self.rsquared}', file=fp)
                print(f'potential: {self.ans_potential}', file=fp)
                """
                print(
                    f'pressure range: '
                    f'{self.pressure[self.i]}-{self.pressure[self.j]}',
                    file=fp
                )
                """
            else:
                print(
                    f'No valid volumes found for {self.material} with '
                    f'exponent of {self.exp}\n'
                    f'Try relaxing filter parameters or different '
                    f'exponent',
                    file=fp
                )

        if self.has_valid_volumes:
            fig, ax = plt.subplots(1, 1)
            dra_plot(
                self.log_v, self.log_p_exp,
                minimum=self.i, maximum=self.j,
                slope=self.fit_grad[self.i, self.j],
                intercept=self.fit_intercept[self.i, self.j],
                exp=self.exp,
                ax=ax,
            )
            plt.savefig(
                filepath / 'optimum_plot.png',
                bbox_inches='tight'
            )
            if verbose:
                plt.show()
            plt.close()


def analyseDA(
    isotherm,
    optimum_criteria='max_points',
    material=None,
    exp=None,
    output_dir=None,
    verbose=False,
    **kwargs,
):
    if output_dir is None:
        output_dir = './dubinin/'

    if material is None:
        material = Path(str(isotherm.material))

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    output_subdir = output_dir / material
    output_subdir.mkdir(exist_ok=True, parents=True)

    result = DubininResult(
        isotherm,
        exp=exp,
        **kwargs,
    )
    filtered = DubininFilteredResults(
        result,
        optimum_criteria=optimum_criteria,
        verbose=verbose,
        **kwargs
    )
    filtered.export(output_subdir, verbose=verbose)

    return filtered


def analyseDR(
    isotherm,
    optimum_criteria='max_points',
    output_dir=None,
    verbose=False,
    **kwargs,
):
    return analyseDA(
        isotherm,
        optimum_criteria=optimum_criteria,
        exp=2,
        output_dir=output_dir,
        verbose=verbose,
        **kwargs,
    )


if __name__ == "__main__":
    """
    for testing
    """
    import glob
    inPath = '../aif/'
    for file in glob.glob(f'{inPath}*.aif'):
        print(file)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            isotherm = pgp.isotherm_from_aif(file)
        dub = analyseDR(
            isotherm, verbose=False,
            output_dir='../example_result/DR/',
        )
        if dub.has_valid_volumes:
            print(dub.k)
            print(dub.ans_potential, dub.characteristic_energy)
            print(dub.opt_point_count)
        dub = analyseDA(
            isotherm, verbose=False,
            output_dir='../example_result/DA/',
        )
        if dub.has_valid_volumes:
            print(dub.k)
            print(dub.ans_potential, dub.characteristic_energy)
            print(dub.opt_point_count)
