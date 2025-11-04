"""Rational sigmoid model for generic curve fitting."""

import numpy as np
from phytorch.models.base import Model


class Sigmoidal(Model):
    """Rational sigmoid curve model.

    Model equation:
        y(x) = ymax / (1 + |x/x50|^s)

    where ymax is the maximum value, x50 is the x value at half-maximum,
    and s is the steepness parameter.

    This is a general-purpose sigmoidal curve useful for many response
    curves in plant physiology.

    Reference:
        TODO: Add proper citation
    """

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute sigmoidal response.

        Args:
            data: {'x': independent variable}
            parameters: {
                'ymax': maximum y value,
                'x50': x value at half-maximum,
                's': steepness parameter
            }

        Returns:
            Predicted y values
        """
        x = np.asarray(data['x'])
        ymax = parameters['ymax']
        x50 = parameters['x50']
        s = parameters['s']

        return ymax / (1 + np.abs(x / x50) ** s)

    def parameter_info(self) -> dict:
        return {
            'ymax': {
                'default': 1.0,
                'bounds': (0.0, np.inf),
                'units': '',
                'description': 'Maximum y value',
                'symbol': 'y_max'
            },
            'x50': {
                'default': 1.0,
                'bounds': (-np.inf, np.inf),
                'units': '',
                'description': 'x value at half-maximum response',
                'symbol': 'x₅₀'
            },
            's': {
                'default': 2.0,
                'bounds': (0.1, 20.0),
                'units': '',
                'description': 'Steepness parameter',
                'symbol': 's'
            }
        }

    def required_data(self) -> list:
        return ['x', 'y']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data.

        Uses data-driven heuristics:
        - ymax: maximum y value
        - x50: x value closest to half-maximum
        - s: default value of 2.0
        """
        x = np.asarray(data['x'])
        y = np.asarray(data['y'])

        # Estimate ymax from maximum y
        ymax_guess = np.max(y)

        # Estimate x50 from x at half-maximum
        half_max = ymax_guess / 2
        idx_half = np.argmin(np.abs(y - half_max))
        x50_guess = x[idx_half]

        # Avoid x50 = 0
        if np.abs(x50_guess) < 1e-6:
            x50_guess = np.mean(x)

        return {
            'ymax': ymax_guess,
            'x50': x50_guess,
            's': 2.0
        }
