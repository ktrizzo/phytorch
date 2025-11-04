"""Rectangular hyperbola (Michaelis-Menten) model."""

import numpy as np
from phytorch.models.base import Model


class RectangularHyperbola(Model):
    """Rectangular hyperbola (Michaelis-Menten) curve model.

    Model equation:
        y(x) = (ymax * x) / (K + x)

    where ymax is the maximum asymptotic value and K is the half-saturation
    constant (x value at half-maximum).

    This is widely used for substrate saturation kinetics, light response curves,
    and other saturating responses in plant physiology.

    Reference:
        TODO: Add proper citation
    """

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute rectangular hyperbola response.

        Args:
            data: {'x': independent variable}
            parameters: {
                'ymax': maximum asymptotic y value,
                'K': half-saturation constant
            }

        Returns:
            Predicted y values
        """
        x = np.asarray(data['x'])
        ymax = parameters['ymax']
        K = parameters['K']

        return (ymax * x) / (K + x)

    def parameter_info(self) -> dict:
        return {
            'ymax': {
                'default': 1.0,
                'bounds': (0.0, np.inf),
                'units': '',
                'description': 'Maximum asymptotic y value',
                'symbol': 'y_max'
            },
            'K': {
                'default': 1.0,
                'bounds': (0.0, np.inf),
                'units': '',
                'description': 'Half-saturation constant (x at y = ymax/2)',
                'symbol': 'K'
            }
        }

    def required_data(self) -> list:
        return ['x', 'y']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data.

        Uses data-driven heuristics:
        - ymax: maximum y value * 1.1 (to account for asymptote)
        - K: x value at half-maximum
        """
        x = np.asarray(data['x'])
        y = np.asarray(data['y'])

        # Estimate ymax from maximum y
        ymax_guess = np.max(y) * 1.1

        # Estimate K from x at half-maximum
        half_max = ymax_guess / 2
        idx_half = np.argmin(np.abs(y - half_max))
        K_guess = x[idx_half]

        # Ensure K is positive
        if K_guess <= 0:
            K_guess = np.median(x[x > 0]) if np.any(x > 0) else 1.0

        return {
            'ymax': ymax_guess,
            'K': K_guess
        }
