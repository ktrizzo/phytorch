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

    def plot(self, data: dict, parameters: dict, show: bool = True, save: str = None):
        """Custom plot for Rectangular hyperbola.

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
        ax.set_title('Rectangular Hyperbola (Michaelis-Menten)', fontsize=13, fontweight='bold')
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
