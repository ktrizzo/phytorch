"""Buckley-Turnbull-Adams (2012) stomatal conductance model."""

import numpy as np
from phytorch.models.base import Model


class BTA2012(Model):
    """Buckley, Turnbull & Adams (2012) stomatal conductance model (Model 4).

    Model equation:
        gs = Em(Q + i0) / (k + bQ + (Q + i0)Ds)

    where gs is stomatal conductance to water vapor (mol m⁻² s⁻¹),
    Em is maximum leaf transpiration rate (mmol m⁻² s⁻¹),
    Q is irradiance/PPFD (μmol m⁻² s⁻¹),
    i0 is dark transpiration parameter equal to α/φ (μmol m⁻² s⁻¹),
    k is a lumped parameter (μmol m⁻² s⁻¹ mmol mol⁻¹),
    b is a lumped parameter (mmol mol⁻¹), and
    Ds is leaf surface vapor pressure saturation deficit (mmol mol⁻¹).

    This is Model 4 from the paper, which groups parameters as:
    Em = K1(ψsoil + πc), k = K1/χφ, b = K1/χα0, and i0 = α/φ.

    The guard cell advantage (α) is approximated as:
    α = αm φ (Q + i0) / (αm + φ Q)

    Reference:
        Buckley, T.N., Turnbull, T.L., & Adams, M.A. (2012). Simple models
        for stomatal conductance derived from a process model: cross-
        validation against sap flux data. Plant, Cell & Environment,
        35(9), 1647-1662.
    """

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute stomatal conductance.

        Args:
            data: {
                'Q': irradiance/PPFD (μmol m⁻² s⁻¹),
                'Ds': leaf surface vapor pressure saturation deficit (mmol mol⁻¹)
            }
            parameters: {
                'Em': maximum leaf transpiration rate (mmol m⁻² s⁻¹),
                'i0': dark transpiration parameter equal to α/φ (μmol m⁻² s⁻¹),
                'k': lumped parameter (μmol m⁻² s⁻¹ mmol mol⁻¹),
                'b': lumped parameter (mmol mol⁻¹)
            }

        Returns:
            Predicted stomatal conductance (mol m⁻² s⁻¹)
        """
        Q = np.asarray(data['Q'])
        Ds = np.asarray(data['Ds'])

        Em = parameters['Em']
        i0 = parameters['i0']
        k = parameters['k']
        b = parameters['b']

        # BTA Model 4: gs = Em(Q + i0) / (k + bQ + (Q + i0)Ds)
        numerator = Em * (Q + i0)
        denominator = k + b * Q + (Q + i0) * Ds

        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-10)

        # Convert from mmol to mol for consistency
        gs = numerator / denominator / 1000.0

        return gs

    def parameter_info(self) -> dict:
        return {
            'Em': {
                'default': 10.0,
                'bounds': (0.1, 50.0),
                'units': 'mmol m⁻² s⁻¹',
                'description': 'Maximum leaf transpiration rate',
                'symbol': 'E_m'
            },
            'i0': {
                'default': 50.0,
                'bounds': (0.0, 300.0),
                'units': 'μmol m⁻² s⁻¹',
                'description': 'Dark transpiration parameter (α/φ)',
                'symbol': 'i_0'
            },
            'k': {
                'default': 1e4,
                'bounds': (0.0, 1e6),
                'units': 'μmol m⁻² s⁻¹ mmol mol⁻¹',
                'description': 'Lumped parameter K1/χφ',
                'symbol': 'k'
            },
            'b': {
                'default': 20.0 / 3.0,
                'bounds': (0.0, 100.0),
                'units': 'mmol mol⁻¹',
                'description': 'Lumped parameter K1/χα0',
                'symbol': 'b'
            }
        }

    def required_data(self) -> list:
        return ['Q', 'Ds', 'gs']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data."""
        gs = np.asarray(data['gs'])
        Q = np.asarray(data['Q'])

        # Em: related to maximum observed conductance
        Em_guess = np.max(gs) * 1000.0 * 2.0  # Convert mol to mmol and scale
        Em_guess = np.clip(Em_guess, 1.0, 50.0)

        # i0: typical value for dark transpiration parameter
        i0_guess = 50.0

        # k: typical lumped parameter value
        k_guess = 1e4

        # b: typical lumped parameter value
        b_guess = 20.0 / 3.0

        return {
            'Em': Em_guess,
            'i0': i0_guess,
            'k': k_guess,
            'b': b_guess
        }

    def plot(self, data: dict, parameters: dict, show: bool = True, save: str = None):
        """Plot stomatal conductance fit results.

        Args:
            data: Data dictionary with 'Q', 'Ds', and 'gs'
            parameters: Fitted parameters
            show: Whether to display the plot (default: True)
            save: Filename to save plot (default: None)
        """
        import matplotlib.pyplot as plt

        # Extract parameters
        Em = parameters['Em']
        i0 = parameters['i0']
        k = parameters['k']
        b = parameters['b']

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: Predicted vs Observed
        predicted = self.forward(data, parameters)
        observed = np.asarray(data['gs'])

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

        ax1.set_xlabel('Observed gs (mol m⁻² s⁻¹)', fontsize=12)
        ax1.set_ylabel('Predicted gs (mol m⁻² s⁻¹)', fontsize=12)
        ax1.set_title('Predicted vs Observed', fontsize=13, fontweight='bold')
        ax1.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax1.transAxes,
                fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, loc='lower right')

        # Right plot: gs vs Q (irradiance)
        Q = np.asarray(data['Q'])

        ax2.scatter(Q, observed, s=100, alpha=0.6, color='black',
                   edgecolors='black', linewidth=0.5, label='Observed', zorder=3)

        # Plot fitted curve
        # Sort by Q for smooth line
        sort_idx = np.argsort(Q)
        ax2.plot(Q[sort_idx], predicted[sort_idx], 'r-', linewidth=2.5,
                label='Model fit', zorder=2)

        ax2.set_xlabel('Irradiance, Q (μmol m⁻² s⁻¹)', fontsize=12)
        ax2.set_ylabel('Stomatal Conductance, gs (mol m⁻² s⁻¹)', fontsize=12)
        ax2.set_title('Stomatal Conductance Response', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)

        # Add parameter info box
        param_text = f"Fitted Parameters:\n"
        param_text += f"Em = {Em:.3f} mmol m⁻² s⁻¹\n"
        param_text += f"i0 = {i0:.3f} μmol m⁻² s⁻¹\n"
        param_text += f"k = {k:.3f}\n"
        param_text += f"b = {b:.4f} mmol mol⁻¹\n\n"
        param_text += f"R² = {r_squared:.4f}"

        ax2.text(0.02, 0.98, param_text, transform=ax2.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return fig
