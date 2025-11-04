"""Rectangular hyperbola model."""

import numpy as np
from phytorch.models.base import Model


class RectangularHyperbola(Model):
    """Rectangular hyperbola curve model.

    Model equation:
        y(x) = (ymax * x) / (x50 + x)

    where ymax is the maximum asymptotic value and x50 is the half-saturation
    constant (x value at half-maximum).

    This saturating hyperbola is commonly found in resource-limited and biological
    processes such as enzyme kinetics (Michaelis-Menten), light response curves,
    and nutrient uptake.

    Reference:
        Michaelis, L., & Menten, M. L. (1913). The kinetics of invertase action.
    """

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute rectangular hyperbola response.

        Args:
            data: {'x': independent variable}
            parameters: {
                'ymax': maximum asymptotic y value,
                'x50': half-saturation constant
            }

        Returns:
            Predicted y values
        """
        x = np.asarray(data['x'])
        ymax = parameters['ymax']
        x50 = parameters['x50']

        return (ymax * x) / (x50 + x)

    def parameter_info(self) -> dict:
        return {
            'ymax': {
                'default': 1.0,
                'bounds': (0.0, np.inf),
                'units': '',
                'description': 'Maximum asymptotic y value',
                'symbol': 'y_max'
            },
            'x50': {
                'default': 1.0,
                'bounds': (0.0, np.inf),
                'units': '',
                'description': 'Half-saturation constant (x at y = ymax/2)',
                'symbol': 'x_{50}'
            }
        }

    def required_data(self) -> list:
        return ['x', 'y']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data.

        Uses data-driven heuristics:
        - ymax: maximum y value * 1.1 (to account for asymptote)
        - x50: x value at half-maximum
        """
        x = np.asarray(data['x'])
        y = np.asarray(data['y'])

        # Estimate ymax from maximum y
        ymax_guess = np.max(y) * 1.1

        # Estimate x50 from x at half-maximum
        half_max = ymax_guess / 2
        idx_half = np.argmin(np.abs(y - half_max))
        x50_guess = x[idx_half]

        # Ensure x50 is positive
        if x50_guess <= 0:
            x50_guess = np.median(x[x > 0]) if np.any(x > 0) else 1.0

        return {
            'ymax': ymax_guess,
            'x50': x50_guess
        }
