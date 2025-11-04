"""Gaussian (normal distribution) curve model."""

import numpy as np
from phytorch.models.base import Model


class Gaussian(Model):
    """Gaussian (bell curve) model.

    Model equation:
        y(x) = a * exp(-((x - μ)²) / (2*σ²))

    where a is the amplitude (height at peak), μ is the mean (center),
    and σ is the standard deviation (width).

    This model describes bell-shaped responses common in many biological
    processes, such as temperature optima, resource allocation curves,
    and phenological patterns.

    Reference:
        Standard Gaussian distribution
    """

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute Gaussian response.

        Args:
            data: {'x': independent variable}
            parameters: {
                'a': amplitude (height at peak),
                'mu': mean (center of distribution),
                'sigma': standard deviation (width)
            }

        Returns:
            Predicted y values
        """
        x = np.asarray(data['x'])
        a = parameters['a']
        mu = parameters['mu']
        sigma = parameters['sigma']

        # Gaussian function
        y = a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        return y

    def parameter_info(self) -> dict:
        return {
            'a': {
                'default': 1.0,
                'bounds': (0.0, np.inf),
                'units': '',
                'description': 'Amplitude (height at peak)',
                'symbol': 'a'
            },
            'mu': {
                'default': 0.0,
                'bounds': (-np.inf, np.inf),
                'units': '',
                'description': 'Mean (center of distribution)',
                'symbol': 'μ'
            },
            'sigma': {
                'default': 1.0,
                'bounds': (0.0, np.inf),
                'units': '',
                'description': 'Standard deviation (width)',
                'symbol': 'σ'
            }
        }

    def required_data(self) -> list:
        return ['x', 'y']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data.

        Uses data-driven heuristics:
        - a: maximum y value
        - mu: x value at maximum y
        - sigma: estimated from width at half-maximum
        """
        x = np.asarray(data['x'])
        y = np.asarray(data['y'])

        # Estimate amplitude and mean from maximum
        idx_max = np.argmax(y)
        a_guess = y[idx_max]
        mu_guess = x[idx_max]

        # Estimate sigma from half-width at half-maximum
        half_max = a_guess / 2
        # Find points closest to half-maximum on either side of peak
        left_half = np.where((x < mu_guess) & (y > half_max * 0.9) & (y < half_max * 1.1))[0]
        right_half = np.where((x > mu_guess) & (y > half_max * 0.9) & (y < half_max * 1.1))[0]

        if len(left_half) > 0 and len(right_half) > 0:
            # FWHM = 2.355 * sigma for Gaussian
            fwhm = x[right_half[0]] - x[left_half[-1]]
            sigma_guess = fwhm / 2.355
        else:
            # Fallback: estimate from data range
            sigma_guess = (np.max(x) - np.min(x)) / 6

        # Ensure sigma is positive
        if sigma_guess <= 0:
            sigma_guess = 1.0

        # Ensure amplitude is positive
        if a_guess <= 0:
            a_guess = np.mean(y[y > 0]) if np.any(y > 0) else 1.0

        return {
            'a': a_guess,
            'mu': mu_guess,
            'sigma': sigma_guess
        }

    def plot(self, data: dict, parameters: dict, show: bool = True, save: str = None):
        """Custom plot for Gaussian distribution.

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
        ax.set_title('Gaussian (Bell Curve) Distribution', fontsize=13, fontweight='bold')
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
