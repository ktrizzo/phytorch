"""Nonrectangular hyperbola model for light response curves."""

import numpy as np
from phytorch.models.base import Model


class NonrectangularHyperbola(Model):
    """Nonrectangular hyperbola curve model.

    Model equation:
        y(x) = (1/(2θ)) * (αx + ymax - sqrt((αx + ymax)² - 4θαxymax))

    where α is the initial slope (quantum yield), ymax is the maximum
    asymptotic value, and θ is the curvature parameter (convexity).

    This model is commonly used for photosynthetic light response curves,
    allowing for a smooth transition between the light-limited and
    light-saturated regions.

    Reference:
        TODO: Add proper citation
    """

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute nonrectangular hyperbola response.

        Args:
            data: {'x': independent variable (e.g., light intensity)}
            parameters: {
                'alpha': initial slope (quantum yield),
                'ymax': maximum asymptotic y value,
                'theta': curvature parameter (0 < θ < 1)
            }

        Returns:
            Predicted y values
        """
        x = np.asarray(data['x'])
        alpha = parameters['alpha']
        ymax = parameters['ymax']
        theta = parameters['theta']

        # Calculate terms
        ax_ymax = alpha * x + ymax

        # Ensure discriminant is non-negative
        discriminant = ax_ymax ** 2 - 4 * theta * alpha * x * ymax
        discriminant = np.maximum(discriminant, 0)

        # Compute response
        y = (ax_ymax - np.sqrt(discriminant)) / (2 * theta)

        return y

    def parameter_info(self) -> dict:
        return {
            'alpha': {
                'default': 0.5,
                'bounds': (0.0, 1.0),
                'units': '',
                'description': 'Initial slope (quantum yield)',
                'symbol': 'α'
            },
            'ymax': {
                'default': 1.0,
                'bounds': (0.0, np.inf),
                'units': '',
                'description': 'Maximum asymptotic y value',
                'symbol': 'y_max'
            },
            'theta': {
                'default': 0.7,
                'bounds': (0.01, 0.99),
                'units': '',
                'description': 'Curvature parameter (convexity)',
                'symbol': 'θ'
            }
        }

    def required_data(self) -> list:
        return ['x', 'y']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data.

        Uses data-driven heuristics:
        - ymax: maximum y value * 1.05
        - alpha: initial slope from first few points
        - theta: typical value of 0.7
        """
        x = np.asarray(data['x'])
        y = np.asarray(data['y'])

        # Estimate ymax from maximum y
        ymax_guess = np.max(y) * 1.05

        # Estimate alpha from initial slope
        # Use first 20% of data sorted by x
        sorted_idx = np.argsort(x)
        n_init = max(2, int(len(x) * 0.2))
        x_init = x[sorted_idx[:n_init]]
        y_init = y[sorted_idx[:n_init]]

        if len(x_init) > 1 and x_init[-1] > x_init[0]:
            alpha_guess = (y_init[-1] - y_init[0]) / (x_init[-1] - x_init[0])
            alpha_guess = np.clip(alpha_guess, 0.01, 0.9)
        else:
            alpha_guess = 0.5

        return {
            'alpha': alpha_guess,
            'ymax': ymax_guess,
            'theta': 0.7
        }

    def plot(self, data: dict, parameters: dict, show: bool = True, save: str = None):
        """Custom plot for Nonrectangular hyperbola.

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
        ax.set_title('Nonrectangular Hyperbola', fontsize=13, fontweight='bold')
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
