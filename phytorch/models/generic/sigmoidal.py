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

    def plot(self, data: dict, parameters: dict, show: bool = True, save: str = None):
        """Custom plot for Sigmoidal curve.

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
        ax.set_title('Rational Sigmoidal Curve', fontsize=13, fontweight='bold')
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
