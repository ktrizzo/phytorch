"""Peaked Arrhenius equation for temperature response with high-temperature deactivation."""

import numpy as np
from phytorch.models.base import Model


class PeakedArrhenius(Model):
    """Peaked Arrhenius equation for temperature response with deactivation.

    Model equation:
        y(x) = ymax * f_arr(x) * f_peak(x)

    where:
        f_arr(x) = exp(Ha/(R*T_ref) - Ha/(R*x))
        f_peak(x) = (1 + exp(Hd/R * (1/Topt - 1/T_ref) - log(Hd/Ha - 1))) /
                    (1 + exp(Hd/R * (1/Topt - 1/x) - log(Hd/Ha - 1)))

    Parameters:
        - x: temperature (K)
        - ymax: maximum value at optimum temperature
        - Ha: activation energy (kJ/mol)
        - Hd: deactivation energy (kJ/mol)
        - Topt: optimum temperature (K)
        - R: gas constant = 0.008314 kJ/(mol·K) (fixed)
        - T_ref: reference temperature = 298.15 K (25°C, fixed)

    This model describes enzyme and metabolic temperature responses that
    increase exponentially at low temperatures but decline at high temperatures
    due to denaturation or other deactivation processes.

    Reference:
        TODO: Add proper citation
    """

    def __init__(self):
        super().__init__()
        self.R = 0.008314  # kJ/(mol·K)
        self.T_ref = 298.15  # K (25°C)

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute peaked Arrhenius temperature response.

        Args:
            data: {'x': temperature (K)}
            parameters: {
                'ymax': maximum value at optimum temperature,
                'Ha': activation energy (kJ/mol),
                'Hd': deactivation energy (kJ/mol),
                'Topt': optimum temperature (K)
            }

        Returns:
            Predicted y values
        """
        x = np.asarray(data['x'])
        ymax = parameters['ymax']
        Ha = parameters['Ha']
        Hd = parameters['Hd']
        Topt = parameters['Topt']

        # Arrhenius component
        f_arr = np.exp(Ha / (self.R * self.T_ref) - Ha / (self.R * x))

        # Peak/deactivation component
        Hd_Ha_ratio = Hd / Ha
        # Ensure ratio is > 1 to avoid log of negative number
        Hd_Ha_ratio = np.maximum(Hd_Ha_ratio, 1.0001)
        log_term = np.log(Hd_Ha_ratio - 1)

        numerator = 1 + np.exp(Hd / self.R * (1 / Topt - 1 / self.T_ref) - log_term)
        denominator = 1 + np.exp(Hd / self.R * (1 / Topt - 1 / x) - log_term)

        f_peak = numerator / denominator

        # Final response
        y = ymax * f_arr * f_peak

        return y

    def parameter_info(self) -> dict:
        return {
            'ymax': {
                'default': 1.0,
                'bounds': (0.0, np.inf),
                'units': '',
                'description': 'Maximum value at optimum temperature',
                'symbol': 'y_max'
            },
            'Ha': {
                'default': 50.0,
                'bounds': (0.0, 150.0),
                'units': 'kJ/mol',
                'description': 'Activation energy',
                'symbol': 'H_a'
            },
            'Hd': {
                'default': 200.0,
                'bounds': (150.0, 400.0),
                'units': 'kJ/mol',
                'description': 'Deactivation energy',
                'symbol': 'H_d'
            },
            'Topt': {
                'default': 311.15,
                'bounds': (273.15, 333.15),
                'units': 'K',
                'description': 'Optimum temperature',
                'symbol': 'T_opt'
            }
        }

    def required_data(self) -> list:
        return ['x', 'y']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data.

        Uses data-driven heuristics:
        - ymax: maximum y value
        - Topt: temperature at maximum y
        - Ha: typical value of 50 kJ/mol
        - Hd: typical value of 200 kJ/mol
        """
        x = np.asarray(data['x'])
        y = np.asarray(data['y'])

        # Estimate ymax and Topt from maximum
        idx_max = np.argmax(y)
        ymax_guess = y[idx_max]
        Topt_guess = x[idx_max]

        # Ensure Topt is within reasonable bounds
        Topt_guess = np.clip(Topt_guess, 273.15, 333.15)

        # Ensure ymax is positive
        if ymax_guess <= 0:
            ymax_guess = np.mean(y[y > 0]) if np.any(y > 0) else 1.0

        return {
            'ymax': ymax_guess,
            'Ha': 50.0,
            'Hd': 200.0,
            'Topt': Topt_guess
        }

    def plot(self, data: dict, parameters: dict, show: bool = True, save: str = None):
        """Custom plot for Peaked Arrhenius temperature response.

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
        ax.set_title('Peaked Arrhenius Temperature Response', fontsize=13, fontweight='bold')
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
