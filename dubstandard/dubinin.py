import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy import optimize, stats, constants
import numpy as np
from collections import OrderedDict
import pandas as pd
import os
from pprint import pprint
from pathlib import Path

import pygaps.parsing as pgp
import pygaps.characterisation as pgc
import pygaps.graphing.calc_graphs as pgraph
from pygaps.utilities.exceptions import CalculationError


def log_p_exp(
    pressure,
    exp
):
    r"""
    Converts pressure array to log10 of array, raised to exponent
    """
    return (-np.log10(pressure))**exp


class DubininResult:
    """
    Store all possible Dubinin results from an isotherm,
    as well as curvature, and other information for filtering
    later.
    """
    def __init__(
        self,
        isotherm,
        exp: float = None,
        **kwargs,
    ):
        self.material = isotherm.material
        self.adsorbate = isotherm.adsorbate
        self.iso_temp = isotherm.temperature
        self.molar_mass = self.adsorbate.molar_mass()
        self.liquid_density = self.adsorbate.liquid_density(self.iso_temp)

        self.pressure = isotherm.pressure(
            branch='ads',
            pressure_mode='relative',
        )
        self.loading = isotherm.loading_at(
            pressure=list(self.pressure),
            branch='ads',
            pressure_mode='relative',
            loading_unit='mol',
            loading_basis='molar',
            material_unit='g',
            material_basis='mass',
        )

        self.total_pore_volume = max(self.loading) * self.molar_mass / self.liquid_density

        self.log_v = pgc.dr_da_plots.log_v_adj(
            self.loading,
            self.molar_mass,
            self.liquid_density,
        )

        bounds = kwargs.get('bounds', [1, 3])

        def dr_fit(exp, ret=False):
            slope, intercept, corr_coef, _, stderr = stats.linregress(
                log_p_exp(self.pressure, exp), self.log_v
            )
            if ret:
                return slope, intercept, corr_coef
            return stderr

        if exp is None:
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

        try:
            spline = PchipInterpolator(
                np.flip(self.log_p_exp),
                np.flip(self.log_v),
            )
            second_derivative = spline.derivative(nu=2)
        except ValueError as e:
            print(e)
            pass

        num_points = len(pressure)
        self.result = {}
        for i in range(num_points):
            for j in range(i+3, num_points):
                self.result[i, j] = {}
                pressure_range = [
                    self.pressure[i],
                    self.pressure[j]
                ]

                try:
                    (
                        slope,
                        intercept,
                        corr_coef,
                        p_val,
                        stderr,
                    ) = stats.linregress(
                        self.log_p_exp[i:j],
                        self.log_v[i:j]
                    )
                    microp_volume = 10**intercept
                    potential = (-np.log(10)**(self.exp - 1) *
                                (constants.gas_constant * self.iso_temp)**(self.exp) \
                                 / slope)**(1 / self.exp) / 1000
                    self.result[i, j] = {
                        'microp_volume': microp_volume,
                        'potential': potential,
                        'slope': slope,
                        'intercept': intercept,
                        'corr_coef': corr_coef,
                        'min_p_index': i,
                        'max_p_index': j,
                    }

                    self.result[i, j]['point_count'] = j - i
                    self.result[i, j]['pressure_range'] = pressure_range

                except CalculationError as e:
                    print(e)
                    continue

                log_p_exp = self.log_p_exp[i:j]
                try:
                    derivs = [
                       min(second_derivative(log_p_exp)),
                       max(second_derivative(log_p_exp))
                    ]
                    deriv_change = abs(derivs[1]-derivs[0])
                    x_range = abs(log_p_exp[0]-log_p_exp[-1])
                    relative_change = deriv_change / np.log10(x_range)
                    self.result[i, j]['curvature'] = abs(relative_change)
                except ValueError:
                    continue
                except UnboundLocalError:
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
        **kwargs,
    ):
        self.__dict__.update(dubinin_result.__dict__)

        p_limits = kwargs.get('p_limits', [0, 0.1])
        self._filter(
            'pressure_range',
            lambda x: x[0] < p_limits[0] or x[1] > p_limits[1]
        )

        curvature_limit = kwargs.get('curvature_limit', 0.02)
        self._filter(
            'curvature',
            lambda x: abs(x) > curvature_limit
        )

        max_volume = kwargs.get(
            'max_volume', self.total_pore_volume
        )
        self._filter(
            'microp_volume',
            lambda x: x > max_volume
        )

        min_points = kwargs.get('min_points', 5)
        self._filter(
            'point_count',
            lambda x: x < min_points
        )

        min_corr_coef = kwargs.get('min_corr_coef', 0.95)
        self._filter(
            'corr_coef',
            lambda x: abs(x) < min_corr_coef
        )

        self.filter_params = kwargs

        if len(self.result) == 0:
            raise ValueError(
                'No valid results with applied filters'
            )
            return

        self._stats()

        self._optimum(optimum_criteria)

    def _optimum(
        self,
        optimum_criteria,
    ):

        if optimum_criteria not in [
            'min_volume',
            'max_points',
            'max_corr_coef'
        ]:
            raise ValueError(
                f'{optimum_criteria} is invalid selection '
                f'criterion for selection of optimum Dubinin '
                f'volume.'
            )

        if optimum_criteria == 'min_volume':
            optimum = 1e10
            for r in self.result:
                if self.result[r]['microp_volume'] < optimum:
                    optimum = self.result[r]['microp_volume']
                    self.optimum = self.result[r]

        if optimum_criteria == 'max_points':
            optimum = 0
            for r in self.result:
                if self.result[r]['point_count'] > optimum:
                    optimum = self.result[r]['point_count']
                    self.optimum = self.result[r]

        if optimum_criteria == 'max_corr_coef':
            optimum = 0
            for r in self.result:
                if abs(self.result[r]['corr_coef']) > optimum:
                    optimum = self.result[r]['corr_coef']
                    self.optimum = self.result[r]

        self.optimum_criteria = optimum_criteria

    def _filter(
        self,
        key,
        criteria,
    ):
        """
        Remove results based on filter criteria. Criteria given
        in the form of lambda expression.
        """
        if len(self.result) == 0:
            print(
                f'{self.material}:\n'
                f'Nothing to filter'
            )
        to_remove = []
        for r in self.result:
            try:
                result = self.result[r]
                if criteria(result[key]):
                    to_remove.append(r)
            except KeyError:
                pass

        for r in to_remove:
            del self.result[r]

    def _sort(self):
        """
        Sort results (assumedly filtered) by `point_count` and
        `corr_coef`.
        TODO: make output in a reasonable format.
        """
        self.result = OrderedDict(sorted(
            self.result.items(),
            key=lambda x: (x[1]['point_count'], x[1]['corr_coef'])
        )
        )

    def _stats(self):
        """
        Determines mean and standard deviation of filtered volumes.
        """
        volumes = [
            x['microp_volume'] for x in self.result.values()
        ]
        self.mean = np.mean(volumes)
        self.stddev = np.std(volumes)

    def export(
        self,
        filepath,
        verbose=False,
    ):
        """
        Exports summary of results
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        with (Path(f'{filepath}filter_summary.json')).open('w') as fp:
            pprint(self.filter_params, fp)

        with (Path(f'{filepath}optimum.txt')).open('w') as fp:
            print(
                f"Optimum volume selected using {self.optimum_criteria}: \n"
                f"Pore volume: {self.optimum['microp_volume']}\n"
                f"Points: {self.optimum['point_count']}\n"
                f"Potential: {self.optimum['potential']}\n"
                f"Slope: {self.optimum['slope']}\n"
                f"Intercept: {self.optimum['intercept']}\n"
                f"Correlation coefficent: {self.optimum['corr_coef']}\n"
                f"Exponent: {self.exp}\n"
                f"Pressure range: "
                f"{self.optimum['pressure_range'][0]} - "
                f"{self.optimum['pressure_range'][1]}\n",
                file=fp)

        pd.DataFrame(self.result).transpose().to_csv(
            f'{filepath}filtered_results.csv',
            index=False
        )

        fig, ax = plt.subplots(1, 1)
        """
        pgraph.dra_plot(
            logv=self.log_v,
            log_n_p0p=self.log_p_exp,
            minimum=self.optimum['min_p_index'],
            maximum=self.optimum['max_p_index'],
            slope=self.optimum['slope'],
            intercept=self.optimum['intercept'],
            exp=self.exp,
            ax=ax,
        )
        """
        ax.scatter(
            self.log_p_exp,
            self.log_v,
            label='all data',
            ec='grey', fc='none',
            marker='o',
        )

        idxmin = self.optimum['min_p_index']
        idxmax = self.optimum['max_p_index']
        ax.scatter(
            self.log_p_exp[idxmin:idxmax],
            self.log_v[idxmin:idxmax],
            color='r', marker='o',
            label='fit data',
        )


        x = np.linspace(
            min(self.log_p_exp),
            max(self.log_p_exp),
            100
        )
        y = self.optimum['slope'] * x + self.optimum['intercept']

        ax.plot(x, y, label='fit')
        ax.legend(
            title=f'exp: {self.exp:.2f}'
        )

        if verbose:
            plt.show()
        fig.savefig(f'{filepath}optimum.png')


def analyseDA(
    isotherm,
    optimum_criteria='max_points',
    exp=None,
    output_dir=None,
    verbose=False,
    **kwargs,
):
    if output_dir is None:
        output_dir = './dubinin/'

    output_subdir = f'{output_dir}{isotherm.material}/'

    result = DubininResult(
        isotherm,
        exp=exp,
        **kwargs,
    )
    filtered = DubininFilteredResults(
        result,
        optimum_criteria=optimum_criteria,
        **kwargs
    )

    return filtered


def analyseDR(
    isotherm,
    optimum_criteria='max_points',
    output_dir=None,
    verbose=False,
    **kwargs,
):
    analyseDA(
        isotherm,
        optimum_criteria=optimum_criteria,
        exp=2,
        output_dir=output_dir,
        verbose=verbose,
        **kwargs,
    )


if __name__ == "__main__":
    inPath = '/home/pcxtsbl/CodeProjects/labcore_upload/robert/aif/'
    file = 'ACC2600.aif'
    isotherm = pgp.isotherm_from_aif(f'{inPath}{file}')
    analyseDR(isotherm, verbose=True,) 
