"""Linear regression model."""

import numpy as np
from phytorch.models.base import Model


class Linear(Model):
    """Linear regression model.

    Model equation:
        y(x) = a + b*x

    where a is the intercept and b is the slope.

    This is the simplest regression model, useful for linear relationships
    or as a baseline comparison for more complex models.

    Reference:
        Standard linear regression
    """

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute linear response.

        Args:
            data: {'x': independent variable}
            parameters: {
                'a': intercept,
                'b': slope
            }

        Returns:
            Predicted y values
        """
        x = np.asarray(data['x'])
        a = parameters['a']
        b = parameters['b']

        return a + b * x

    def parameter_info(self) -> dict:
        return {
            'a': {
                'default': 0.0,
                'bounds': (-np.inf, np.inf),
                'units': '',
                'description': 'Intercept',
                'symbol': 'a'
            },
            'b': {
                'default': 1.0,
                'bounds': (-np.inf, np.inf),
                'units': '',
                'description': 'Slope',
                'symbol': 'b'
            }
        }

    def required_data(self) -> list:
        return ['x', 'y']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data using least squares.

        Computes exact least squares solution for initial guess.
        """
        x = np.asarray(data['x'])
        y = np.asarray(data['y'])

        # Compute least squares solution
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xx = np.sum(x * x)
        sum_xy = np.sum(x * y)

        # Calculate slope and intercept
        denominator = n * sum_xx - sum_x * sum_x

        if np.abs(denominator) > 1e-10:
            b_guess = (n * sum_xy - sum_x * sum_y) / denominator
            a_guess = (sum_y - b_guess * sum_x) / n
        else:
            # Fallback if x has no variance
            a_guess = np.mean(y)
            b_guess = 0.0

        return {
            'a': a_guess,
            'b': b_guess
        }
