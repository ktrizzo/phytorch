"""Beta distribution (PDF form) scaled to arbitrary range."""

import numpy as np
from phytorch.models.base import Model
from scipy.special import beta as beta_function


class Beta(Model):
    """Beta distribution probability density function on arbitrary range.

    Model equation:
        For xmin ≤ x ≤ xmax:
            u = (x - xmin) / (xmax - xmin)  [normalize to [0,1]]
            y(x) = a * u^(α-1) * (1-u)^(β-1) / B(α, β)
        Outside range:
            y(x) = 0

    where:
        - a: amplitude (height scaling factor)
        - xmin, xmax: range of distribution
        - α (alpha): shape parameter controlling left tail
        - β (beta): shape parameter controlling right tail
        - B(α, β): beta function = Γ(α)Γ(β)/Γ(α+β)

    The beta distribution is useful for modeling bounded responses, resource
    allocation patterns, and probability distributions on finite intervals.

    Reference:
        TODO: Add proper citation
    """

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute Beta PDF on arbitrary range.

        Args:
            data: {'x': independent variable}
            parameters: {
                'a': amplitude (height scaling factor),
                'xmin': minimum of range,
                'xmax': maximum of range,
                'alpha': left shape parameter,
                'beta': right shape parameter
            }

        Returns:
            Predicted y values
        """
        x = np.asarray(data['x'])
        a = parameters['a']
        xmin = parameters['xmin']
        xmax = parameters['xmax']
        alpha = parameters['alpha']
        beta_param = parameters['beta']

        # Initialize output
        y = np.zeros_like(x, dtype=float)

        # Only compute for x in [xmin, xmax]
        mask = (x >= xmin) & (x <= xmax)
        if np.any(mask):
            # Normalize to [0, 1]
            u = (x[mask] - xmin) / (xmax - xmin)

            # Clip to avoid numerical issues at boundaries
            u = np.clip(u, 1e-10, 1 - 1e-10)

            # Beta PDF
            B = beta_function(alpha, beta_param)
            y[mask] = a * (u ** (alpha - 1)) * ((1 - u) ** (beta_param - 1)) / B

        return y

    def parameter_info(self) -> dict:
        return {
            'a': {
                'default': 1.0,
                'bounds': (0.0, np.inf),
                'units': '',
                'description': 'Amplitude (height scaling factor)',
                'symbol': 'a'
            },
            'xmin': {
                'default': 0.0,
                'bounds': (-np.inf, np.inf),
                'units': '',
                'description': 'Minimum of range',
                'symbol': 'x_min'
            },
            'xmax': {
                'default': 1.0,
                'bounds': (-np.inf, np.inf),
                'units': '',
                'description': 'Maximum of range',
                'symbol': 'x_max'
            },
            'alpha': {
                'default': 2.0,
                'bounds': (0.1, 10.0),
                'units': '',
                'description': 'Left shape parameter',
                'symbol': 'α'
            },
            'beta': {
                'default': 2.0,
                'bounds': (0.1, 10.0),
                'units': '',
                'description': 'Right shape parameter',
                'symbol': 'β'
            }
        }

    def required_data(self) -> list:
        return ['x', 'y']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data.

        Uses data-driven heuristics:
        - a: maximum y value * 1.2 (to account for beta normalization)
        - xmin, xmax: data range with small margin
        - alpha, beta: estimated from peak location (symmetric if centered)
        """
        x = np.asarray(data['x'])
        y = np.asarray(data['y'])

        # Estimate amplitude
        a_guess = np.max(y) * 1.2

        # Estimate range from data
        x_range = np.max(x) - np.min(x)
        xmin_guess = np.min(x) - x_range * 0.05
        xmax_guess = np.max(x) + x_range * 0.05

        # Estimate alpha and beta from peak location
        idx_max = np.argmax(y)
        x_max = x[idx_max]

        # Normalized peak location
        peak_norm = (x_max - xmin_guess) / (xmax_guess - xmin_guess)
        peak_norm = np.clip(peak_norm, 0.1, 0.9)

        # For beta distribution, mode = (α-1)/(α+β-2) when α,β > 1
        # If peak is at 0.5, use symmetric α = β = 2
        # Otherwise adjust based on skewness
        if peak_norm > 0.4 and peak_norm < 0.6:
            alpha_guess = 2.0
            beta_guess = 2.0
        elif peak_norm < 0.5:
            # Peak is left of center, increase beta
            alpha_guess = 2.0
            beta_guess = 3.0
        else:
            # Peak is right of center, increase alpha
            alpha_guess = 3.0
            beta_guess = 2.0

        return {
            'a': a_guess if a_guess > 0 else 1.0,
            'xmin': xmin_guess,
            'xmax': xmax_guess,
            'alpha': alpha_guess,
            'beta': beta_guess
        }

    def plot(self, data: dict, parameters: dict, show: bool = True, save: str = None):
        """Custom plot for beta distribution.

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
        ax.set_title('Beta Distribution', fontsize=13, fontweight='bold')
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
