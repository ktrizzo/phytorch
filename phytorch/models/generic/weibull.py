"""Weibull distribution (PDF form)."""

import numpy as np
from phytorch.models.base import Model


class Weibull(Model):
    """Weibull distribution probability density function.

    Model equation:
        For x > x0:
            y(x) = ymax * (k/λ) * ((x - x0)/λ)^(k-1) * exp(-((x - x0)/λ)^k)
        For x ≤ x0:
            y(x) = 0

    where:
        - ymax: amplitude (maximum height of distribution)
        - x0: location/threshold parameter (start of distribution)
        - λ (lambda): scale parameter
        - k: shape parameter (k > 1 gives right-skewed, k < 1 gives left-skewed)

    The Weibull distribution is commonly used to model leaf size distributions,
    failure times, and other skewed biological data.

    Reference:
        TODO: Add proper citation
    """

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute Weibull PDF.

        Args:
            data: {'x': independent variable}
            parameters: {
                'ymax': amplitude (maximum height),
                'x0': location/threshold parameter,
                'lambda': scale parameter,
                'k': shape parameter
            }

        Returns:
            Predicted y values
        """
        x = np.asarray(data['x'])
        ymax = parameters['ymax']
        x0 = parameters['x0']
        lam = parameters['lambda']
        k = parameters['k']

        # Shifted x
        x_shifted = x - x0

        # Initialize output
        y = np.zeros_like(x, dtype=float)

        # Only compute for x > x0
        mask = x_shifted > 0
        if np.any(mask):
            z = x_shifted[mask] / lam
            y[mask] = ymax * (k / lam) * (z ** (k - 1)) * np.exp(-(z ** k))

        return y

    def parameter_info(self) -> dict:
        return {
            'ymax': {
                'default': 1.0,
                'bounds': (0.0, np.inf),
                'units': '',
                'description': 'Amplitude (maximum height of distribution)',
                'symbol': 'y_max'
            },
            'x0': {
                'default': 0.0,
                'bounds': (-np.inf, np.inf),
                'units': '',
                'description': 'Location/threshold parameter',
                'symbol': 'x_0'
            },
            'lambda': {
                'default': 1.0,
                'bounds': (0.0, np.inf),
                'units': '',
                'description': 'Scale parameter',
                'symbol': 'λ'
            },
            'k': {
                'default': 2.0,
                'bounds': (0.1, 10.0),
                'units': '',
                'description': 'Shape parameter',
                'symbol': 'k'
            }
        }

    def required_data(self) -> list:
        return ['x', 'y']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data.

        Uses data-driven heuristics:
        - ymax: maximum y value
        - x0: minimum x value where y > 0
        - lambda: estimated from data spread
        - k: typical value of 2.0
        """
        x = np.asarray(data['x'])
        y = np.asarray(data['y'])

        # Estimate ymax
        ymax_guess = np.max(y)

        # Estimate x0 from minimum x where y is significant
        threshold = ymax_guess * 0.05
        x_significant = x[y > threshold]
        if len(x_significant) > 0:
            x0_guess = np.min(x_significant) - (np.max(x) - np.min(x)) * 0.1
        else:
            x0_guess = np.min(x)

        # Estimate lambda from data spread
        x_shifted = x - x0_guess
        x_shifted_pos = x_shifted[x_shifted > 0]
        if len(x_shifted_pos) > 0:
            lambda_guess = np.mean(x_shifted_pos)
        else:
            lambda_guess = (np.max(x) - np.min(x)) / 3

        # Ensure positive
        if lambda_guess <= 0:
            lambda_guess = 1.0

        return {
            'ymax': ymax_guess if ymax_guess > 0 else 1.0,
            'x0': x0_guess,
            'lambda': lambda_guess,
            'k': 2.0
        }

    def plot(self, data: dict, parameters: dict, show: bool = True, save: str = None):
        """Custom plot for Weibull distribution.

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
        ax.set_title('Weibull Distribution', fontsize=13, fontweight='bold')
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
