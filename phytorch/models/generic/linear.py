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

    def plot(self, data: dict, parameters: dict, show: bool = True, save: str = None):
        """Custom plot for Linear regression.

        Args:
            data: Data dictionary with 'x' and 'y'
            parameters: Fitted parameters
            show: Whether to display the plot (default: True)
            save: Filename to save plot (default: None)
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(data['x'], data['y'], s=100, alpha=0.6, color='black',
                   edgecolors='black', linewidth=0.5, label='Observed', zorder=3)

        x_fine = np.linspace(data['x'].min(), data['x'].max(), 200)
        y_fine = self.forward({'x': x_fine}, parameters)
        ax.plot(x_fine, y_fine, 'r-', linewidth=2.5, label='Model fit', zorder=2)

        predicted = self.forward(data, parameters)
        observed = np.asarray(data['y'])
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title('Linear Regression', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        param_text = f"Fitted Parameters:\n"
        for param_name, param_value in parameters.items():
            info = self.parameter_info()[param_name]
            param_text += f"{info['symbol']} = {param_value:.4f} {info['units']}\n"
        param_text += f"\nR² = {r_squared:.4f}"

        ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=200, bbox_inches='tight')
        if show:
            plt.show()

        return fig, ax
