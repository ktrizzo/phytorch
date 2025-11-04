"""Pressure-volume curve model for leaf hydraulic capacitance (Sack, John, Buckley 2018)."""

import numpy as np
from phytorch.models.base import Model


class SJB2018(Model):
    """Pressure-volume curve model for leaf water relations.

    Model equation:
        ψ(w) = p(w) + π(w)

    where:
        p(w) = πₒ · max(0, (w - w_tlp)/(1 - w_tlp))^ε   (turgor pressure, positive MPa)
        π(w) = -πₒ / w                                   (osmotic potential, negative MPa)

    and w is relative water content (0-1), πₒ is osmotic pressure at full turgor (positive),
    w_tlp is relative water content at turgor loss point, and ε is compartmental
    wall elasticity.

    Reference:
        Sack, L., John, G.P., and Buckley, T.N. (2018)
        "ABA Accumulation in Dehydrating Leaves Is Associated with Decline in
        Cell Volume, Not Turgor Pressure"
        Plant Physiology 178(1):258-275
        https://doi.org/10.1104/pp.17.01097
    """

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute water potential from relative water content.

        Args:
            data: {'w': relative water content (unitless, 0-1)}
            parameters: {
                'pi_o': osmotic pressure at full turgor (MPa, positive value),
                'w_tlp': relative water content at turgor loss point (0-1),
                'epsilon': compartmental wall elasticity
            }

        Returns:
            Predicted water potential (MPa, negative values)
        """
        w = np.asarray(data['w'])
        pi_o = parameters['pi_o']
        w_tlp = parameters['w_tlp']
        epsilon = parameters['epsilon']

        # Turgor pressure (positive, MPa)
        p = pi_o * np.maximum(0, (w - w_tlp) / (1 - w_tlp)) ** epsilon

        # Osmotic potential (negative, MPa)
        pi = -pi_o / w

        return p + pi

    def parameter_info(self) -> dict:
        return {
            'pi_o': {
                'default': 2.0,
                'bounds': (0.1, 5.0),
                'units': 'MPa',
                'description': 'Osmotic pressure at full turgor (positive value; osmotic potential = -πₒ)',
                'symbol': 'πₒ'
            },
            'w_tlp': {
                'default': 0.85,
                'bounds': (0.5, 0.99),
                'units': '',
                'description': 'Relative water content at turgor loss point',
                'symbol': 'w_tlp'
            },
            'epsilon': {
                'default': 1.0,
                'bounds': (0.1, 3.0),
                'units': '',
                'description': 'Compartmental wall elasticity',
                'symbol': 'ε'
            }
        }

    def required_data(self) -> list:
        return ['w', 'psi']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data.

        Uses data-driven heuristics:
        - pi_o: absolute value of psi at full saturation (w=1)
        - w_tlp: typical value of 0.85
        - epsilon: typical value of 1.0 (linear elasticity)
        """
        w = np.asarray(data['w'])
        psi = np.asarray(data['psi'])

        # Estimate pi_o from psi at highest w
        idx_max_w = np.argmax(w)
        pi_o_guess = np.abs(psi[idx_max_w])

        # Ensure reasonable value
        if pi_o_guess < 0.1:
            pi_o_guess = 1.0
        elif pi_o_guess > 5.0:
            pi_o_guess = 2.0

        return {
            'pi_o': pi_o_guess,
            'w_tlp': 0.85,  # Typical value
            'epsilon': 1.0  # Linear elasticity
        }

    def plot(self, data: dict, parameters: dict, show: bool = True, save: str = None):
        """Custom plot for pressure-volume curves showing fitted parameters.

        Args:
            data: Data dictionary with 'w' (RWC) and 'psi' (water potential)
            parameters: Fitted parameters
            show: Whether to display the plot (default: True)
            save: Filename to save plot (default: None)
        """
        import matplotlib.pyplot as plt

        # Calculate derived parameters
        pi_o = parameters['pi_o']
        w_tlp = parameters['w_tlp']
        epsilon = parameters['epsilon']

        # Water potential at turgor loss point
        psi_tlp = -pi_o / w_tlp

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: Predicted vs Observed
        predicted = self.forward(data, parameters)
        observed = np.asarray(data['psi'])

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

        ax1.set_xlabel('Observed ψ (MPa)', fontsize=12)
        ax1.set_ylabel('Predicted ψ (MPa)', fontsize=12)
        ax1.set_title('Predicted vs Observed', fontsize=13, fontweight='bold')
        ax1.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax1.transAxes,
                fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, loc='lower right')

        # Right plot: PV Curve
        ax2.scatter(data['w'], data['psi'], s=100, alpha=0.6, color='black',
                   edgecolors='black', linewidth=0.5, label='Observed', zorder=3)

        # Plot fitted curve
        w_fine = np.linspace(data['w'].min(), 1.0, 200)
        psi_fine = self.forward({'w': w_fine}, parameters)
        ax2.plot(w_fine, psi_fine, 'r-', linewidth=2.5, label='Model fit', zorder=2)

        # Mark turgor loss point
        ax2.axvline(w_tlp, color='gray', linestyle='--', linewidth=1.5,
                   alpha=0.6, label=f'TLP (w={w_tlp:.3f})', zorder=1)

        ax2.set_xlabel('Relative Water Content (dimensionless)', fontsize=12)
        ax2.set_ylabel('Water Potential, ψ (MPa)', fontsize=12)
        ax2.set_title('Pressure-Volume Curve', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)

        # Add parameter info box
        param_text = f"Fitted Parameters:\n"
        param_text += f"πₒ = {pi_o:.3f} MPa\n"
        param_text += f"w_tlp = {w_tlp:.3f}\n"
        param_text += f"ε = {epsilon:.3f}\n\n"
        param_text += f"Derived:\n"
        param_text += f"ψ_tlp = {psi_tlp:.3f} MPa"

        ax2.text(0.02, 0.98, param_text, transform=ax2.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=200, bbox_inches='tight')

        if show:
            plt.show()

        return fig, (ax1, ax2)
