"""Arrhenius equation for temperature response."""

import numpy as np
from phytorch.models.base import Model


class Arrhenius(Model):
    """Arrhenius equation for temperature-dependent processes.

    Model equation:
        y(x) = ymax * exp(Ha/(R*T_ref) - Ha/(R*x))

    where x is temperature (K), Ha is activation energy (kJ/mol),
    R is the gas constant (0.008314 kJ/(mol·K)), and T_ref is the
    reference temperature (298.15 K, 25°C).

    This model describes the exponential increase of reaction rates
    with temperature, commonly used for enzyme kinetics and metabolic
    processes in plants.

    Reference:
        TODO: Add proper citation
    """

    def __init__(self):
        super().__init__()
        self.R = 0.008314  # kJ/(mol·K)
        self.T_ref = 298.15  # K (25°C)

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute Arrhenius temperature response.

        Args:
            data: {'x': temperature (K)}
            parameters: {
                'ymax': maximum value at reference temperature,
                'Ha': activation energy (kJ/mol)
            }

        Returns:
            Predicted y values
        """
        x = np.asarray(data['x'])
        ymax = parameters['ymax']
        Ha = parameters['Ha']

        # Arrhenius response
        y = ymax * np.exp(Ha / (self.R * self.T_ref) - Ha / (self.R * x))

        return y

    def parameter_info(self) -> dict:
        return {
            'ymax': {
                'default': 1.0,
                'bounds': (0.0, np.inf),
                'units': '',
                'description': 'Maximum value at reference temperature (25°C)',
                'symbol': 'y_max'
            },
            'Ha': {
                'default': 50.0,
                'bounds': (0.0, 200.0),
                'units': 'kJ/mol',
                'description': 'Activation energy',
                'symbol': 'H_a'
            }
        }

    def required_data(self) -> list:
        return ['x', 'y']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data.

        Uses data-driven heuristics:
        - ymax: y value closest to reference temperature (298.15 K)
        - Ha: typical value of 50 kJ/mol
        """
        x = np.asarray(data['x'])
        y = np.asarray(data['y'])

        # Find y at temperature closest to 25°C (298.15 K)
        idx_ref = np.argmin(np.abs(x - self.T_ref))
        ymax_guess = y[idx_ref]

        # Ensure positive
        if ymax_guess <= 0:
            ymax_guess = np.mean(y[y > 0]) if np.any(y > 0) else 1.0

        return {
            'ymax': ymax_guess,
            'Ha': 50.0
        }

    def plot(self, data: dict, parameters: dict, show: bool = True, save: str = None):
        """Custom plot for Arrhenius temperature response.

        Args:
            data: Data dictionary with 'x' and 'y'
            parameters: Fitted parameters
            show: Whether to display the plot (default: True)
            save: Filename to save plot (default: None)
        """
        import matplotlib.pyplot as plt

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data
        ax.scatter(data['x'], data['y'], s=100, alpha=0.6, color='black',
                   edgecolors='black', linewidth=0.5, label='Observed', zorder=3)

        # Plot fitted curve
        x_fine = np.linspace(data['x'].min(), data['x'].max(), 200)
        y_fine = self.forward({'x': x_fine}, parameters)
        ax.plot(x_fine, y_fine, 'r-', linewidth=2.5, label='Model fit', zorder=2)

        # Calculate R²
        predicted = self.forward(data, parameters)
        observed = np.asarray(data['y'])
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Labels and title
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title('Arrhenius Temperature Response', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # Add parameter info box
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
