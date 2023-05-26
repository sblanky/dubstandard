from scipy.interpolate import PchipInterpolator
from scipy.signal import argrelextrema
from scipy import optimize, stats, constants
import numpy as np

import warnings

import pygaps.parsing as pgp
import pygaps.characterisation as pgc
from pygaps.utilities.exceptions import CalculationError


def log_p_exp(
    pressure,
    exp
):
    r"""
    Converts pressure array to log of array, raised to exponent
    """
    return (-np.log(pressure))**exp


def sq_log_potential_from_spline(
    log_p_exp,
    log_v,
    total_pore_volume,
    temperature
):
    log_RT = np.log(constants.gas_constant * temperature)
    relative_log_volume = log_v - np.log(total_pore_volume)
    return PchipInterpolator(
        log_p_exp,
        (relative_log_volume / log_p_exp) / log_RT
    )


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
        def zero_matrix(dim):
            return np.zeros([dim, dim])

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

        plateau_pressure = 0.9
        if max(isotherm.pressure()) < plateau_pressure:
            plateau_pressure = max(isotherm.pressure())
        self.plateau_loading = isotherm.loading_at(
            pressure=plateau_pressure,
            branch='ads',
            pressure_mode='relative',
            loading_unit='mol',
            loading_basis='molar',
            material_unit='g',
            material_basis='mass',
        )
        self.total_pore_volume = (
            self.plateau_loading * self.molar_mass /
            self.liquid_density
        )

        self.log_v = pgc.dr_da_plots.log_v_adj(
            self.loading,
            self.molar_mass,
            self.liquid_density
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
        if (
            ultrarouq_knee_idx.size == 0 or
            ultrarouq_knee_idx >= self.rouq_knee_idx
        ):
            ultrarouq_knee_idx = np.array([0])

        ultrarouq_knee_idx = ultrarouq_knee_idx[
            ultrarouq_knee_idx < self.rouq_knee_idx
        ]

        self.ultrarouq_knee_idx = ultrarouq_knee_idx[-1]

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

        try:
            spline = PchipInterpolator(
                np.flip(self.log_p_exp),
                np.flip(self.log_v),
            )
            self.spline_log_p_exp = np.linspace(
                min(self.log_p_exp),
                max(self.log_p_exp),
                1000
            )
            self.spline_log_v = spline(self.spline_log_p_exp)
            self.spline_log_sq_potential = sq_log_potential_from_spline(
                self.spline_log_p_exp, self.spline_log_v,
                self.total_pore_volume, self.iso_temp,
            )
        except ValueError as e:
            print(e)
            pass

        def distance_to_pchip(_s, _log_sq_potential):
            return (
                spline.__call__(_s, nu=0, extrapolate=None)
                - _log_sq_potential
            )

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

                    self.pore_volume[i, j] = np.exp(fit_intercept)
                    if not np.isfinite(self.pore_volume[i, j]):
                        self.pore_volume[i, j] = 0

                    self.potentials[i, j] = (
                        (constants.gas_constant * self.iso_temp) /
                        (-fit_grad)**(1 / self.exp) / 1000
                    )
                    if not np.isfinite(self.potentials[i, j]):
                        self.potentials[i, j] = 0
                    self.log_sq_potential[i, j] = 2*np.log(self.potentials[i, j])

                    try:
                        opt_res_pchip = optimize.minimize(
                            fun=distance_to_pchip,
                            x0=self.log_p_exp[i:j],
                            args=self.log_sq_potential[i,j]
                        )
                        self.pchip_log_p_exp = opt_res_pchip.x[0]

                    except ValueError as e:
                        pass
                    except NameError as e:
                        pass

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
        **kwargs,
    ):
        self.__dict__.update(dubinin_result.__dict__)
        self.filter_params = kwargs

        filter_mask = np.ones_like(dubinin_result.point_count)

        min_points = kwargs.get('min_points', 10)
        filter_mask = filter_mask * (dubinin_result.point_count > min_points)

        min_r2 = kwargs.get('min_r2', 0.9)
        filter_mask = filter_mask * (dubinin_result.fit_rsquared > min_r2)

        filter_mask = filter_mask * dubinin_result.rouq_expand

        max_volume = kwargs.get('max_volume', dubinin_result.total_pore_volume)
        filter_mask = filter_mask * (dubinin_result.pore_volume < max_volume)

        self.has_valid_volumes = False
        if np.sum(filter_mask) != 0:
            self.has_valid_volumes = True
            self.pore_volume_filtered = self.pore_volume * filter_mask
            self.valid_indices = np.where(
                self.pore_volume_filtered > 0
            )

            self.num_valid = len(self.valid_indices[0])
            self.valid_volumes = self.pore_volume_filtered[self.valid_indices]
            self.valid_point_counts = self.point_count[self.valid_indices]
            self.valid_rsquared = self.fit_rsquared[self.valid_indices]

            self.stdev_volume = np.std(self.valid_volumes)

            self.potential_filtered = self.potentials * filter_mask
            self.valid_potentials = self.potential_filtered[self.valid_indices]
            self.stdev_potentials = np.std(self.valid_potentials)

            self._optimise_median_potential()

    def _optimise_median_potential(self):
        median_potential = np.median(self.valid_potentials)
        if median_potential not in self.valid_potentials:
            median_potential = min(
                self.valid_potentials,
                key=lambda x: (median_potential-x)
            )

        indeces = np.where(self.potentials == median_potential)
        median_i = int(indeces[0])
        median_j = int(indeces[1])

        self.i = median_i
        self.j = median_j

        self.volume = self.pore_volume_filtered[median_i, median_j]
        self.rsquared = self.fit_rsquared[median_i, median_j]
        self.ans_potential = self.potentials[median_i, median_j]


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
    return analyseDA(
        isotherm,
        optimum_criteria=optimum_criteria,
        exp=2,
        output_dir=output_dir,
        verbose=verbose,
        **kwargs,
    )


if __name__ == "__main__":
    import glob
    inPath = '../example/aif/'
    for file in glob.glob(f'{inPath}*.aif'):
        print(file)
        isotherm = pgp.isotherm_from_aif(file)
        print(analyseDR(isotherm, verbose=True,).has_valid_volumes)
        print(analyseDA(isotherm, verbose=True,).has_valid_volumes)
