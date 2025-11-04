"""Sigmoidal (rational sigmoid) hydraulic conductance vulnerability curve model."""

import numpy as np
from phytorch.models.base import Model


class Sigmoidal(Model):
    """Sigmoidal (rational sigmoid) hydraulic conductance vulnerability curve.

    Model equation:
        K(ψ) = Kmax / (1 + |ψ/ψ50|^s)

    where Kmax is the maximum hydraulic conductance, ψ50 is the water
    potential giving half of Kmax, and s controls the steepness of the
    vulnerability curve.

    Reference:
        TODO: Add proper citation
    """

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute conductance from water potential.

        Args:
            data: {'psi': water potential (MPa, negative values)}
            parameters: {
                'Kmax': maximum conductance,
                'psi50': P50 value (MPa),
                's': steepness parameter
            }

        Returns:
            Predicted conductance values (same units as Kmax)
        """
        psi = np.asarray(data['psi'])
        Kmax = parameters['Kmax']
        psi50 = parameters['psi50']
        s = parameters['s']

        return Kmax / (1 + np.abs(psi / psi50) ** s)

    def parameter_info(self) -> dict:
        return {
            'Kmax': {
                'default': 10.0,
                'bounds': (0.0, np.inf),
                'units': 'mmol m⁻² s⁻¹ MPa⁻¹',
                'description': 'Maximum hydraulic conductance',
                'symbol': 'K_max'
            },
            'psi50': {
                'default': -1.5,
                'bounds': (-10.0, 0.0),
                'units': 'MPa',
                'description': 'Water potential at 50% loss of conductance',
                'symbol': 'ψ₅₀'
            },
            's': {
                'default': 2.0,
                'bounds': (0.1, 20.0),
                'units': '',
                'description': 'Steepness of vulnerability curve',
                'symbol': 's'
            }
        }

    def required_data(self) -> list:
        return ['psi', 'K']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data.

        Uses data-driven heuristics:
        - Kmax: slightly above max observed conductance
        - psi50: water potential where K ≈ Kmax/2
        - s: typical value of 2.0
        """
        K = np.asarray(data['K'])
        psi = np.asarray(data['psi'])

        Kmax_guess = np.max(K) * 1.1  # Slightly above max observed

        # Find psi where K is closest to Kmax/2
        K_max = np.max(K)
        psi50_guess = psi[np.argmin(np.abs(K - K_max/2))]

        # Ensure psi50 is negative
        if psi50_guess > 0:
            psi50_guess = -1.5

        return {
            'Kmax': Kmax_guess,
            'psi50': psi50_guess,
            's': 2.0  # Typical value
        }

    def plot(self, data: dict, parameters: dict, show: bool = True, save: str = None):
        """Custom plot for vulnerability curves showing fitted parameters.

        Args:
            data: Data dictionary with 'psi' (water potential) and 'K' (conductance)
            parameters: Fitted parameters
            show: Whether to display the plot (default: True)
            save: Filename to save plot (default: None)
        """
        import matplotlib.pyplot as plt

        # Extract parameters
        Kmax = parameters['Kmax']
        psi50 = parameters['psi50']
        s = parameters['s']

        # Calculate derived parameters
        # ψ₁₂: water potential at 12% loss (88% remaining conductance)
        # ψ₈₈: water potential at 88% loss (12% remaining conductance)
        psi12 = psi50 / (2 ** (1/s))  # K = 0.88 * Kmax
        psi88 = psi50 / (0.25 ** (1/s))  # K = 0.12 * Kmax

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: Predicted vs Observed
        predicted = self.forward(data, parameters)
        observed = np.asarray(data['K'])

        ax1.scatter(observed, predicted, s=100, alpha=0.6, color='black',
                   edgecolors='black', linewidth=0.5, zorder=3)

        # 1:1 line
        min_val = min(observed.min(), predicted.min())
        max_val = max(observed.max(), predicted.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--',
                linewidth=1, alpha=0.5, label='1:1 line', zorder=1)

        # Calculate R²
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        ax1.set_xlabel('Observed K (mmol m⁻² s⁻¹ MPa⁻¹)', fontsize=12)
        ax1.set_ylabel('Predicted K (mmol m⁻² s⁻¹ MPa⁻¹)', fontsize=12)
        ax1.set_title('Predicted vs Observed', fontsize=13, fontweight='bold')
        ax1.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax1.transAxes,
                fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, loc='lower right')

        # Right plot: Vulnerability Curve
        ax2.scatter(data['psi'], data['K'], s=100, alpha=0.6, color='black',
                   edgecolors='black', linewidth=0.5, label='Observed', zorder=3)

        # Plot fitted curve
        psi_fine = np.linspace(data['psi'].min(), data['psi'].max(), 200)
        K_fine = self.forward({'psi': psi_fine}, parameters)
        ax2.plot(psi_fine, K_fine, 'r-', linewidth=2.5, label='Model fit', zorder=2)

        # Mark P50 point
        ax2.axvline(psi50, color='gray', linestyle='--', linewidth=1.5,
                   alpha=0.6, label=f'P50 (ψ={psi50:.2f} MPa)', zorder=1)

        ax2.set_xlabel('Water Potential, ψ (MPa)', fontsize=12)
        ax2.set_ylabel('Hydraulic Conductance, K (mmol m⁻² s⁻¹ MPa⁻¹)', fontsize=12)
        ax2.set_title('Vulnerability Curve', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)

        # Add parameter info box
        param_text = f"Fitted Parameters:\n"
        param_text += f"K_max = {Kmax:.3f} mmol m⁻² s⁻¹ MPa⁻¹\n"
        param_text += f"ψ₅₀ = {psi50:.3f} MPa\n"
        param_text += f"s = {s:.3f}\n\n"
        param_text += f"Derived:\n"
        param_text += f"ψ₁₂ = {psi12:.3f} MPa\n"
        param_text += f"ψ₈₈ = {psi88:.3f} MPa"

        ax2.text(0.02, 0.98, param_text, transform=ax2.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=200, bbox_inches='tight')

        if show:
            plt.show()

        return fig, (ax1, ax2)
