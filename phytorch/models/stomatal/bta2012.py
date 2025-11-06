"""Buckley-Turnbull-Adams (2012) stomatal conductance model."""

import numpy as np
from phytorch.models.base import Model


class BTA2012(Model):
    """Buckley, Turnbull & Adams (2012) unified stomatal optimization model.

    Model equation (simplified empirical form):
        gs = gs0 + g1 * A / (Ca * sqrt(VPD))

    where gs is stomatal conductance to water vapor (mol m⁻² s⁻¹),
    gs0 is minimum conductance, g1 is the marginal water use efficiency parameter,
    A is net CO₂ assimilation rate (μmol m⁻² s⁻¹),
    Ca is atmospheric CO₂ concentration (ppm), and
    VPD is vapor pressure deficit (kPa).

    This formulation is based on the unified stomatal optimization theory
    which predicts that stomata operate to maximize carbon gain while
    minimizing water loss.

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
                'A': net CO₂ assimilation rate (μmol m⁻² s⁻¹),
                'VPD': vapor pressure deficit (kPa),
                'Ca': atmospheric CO₂ concentration (ppm, default=400 if not provided)
            }
            parameters: {
                'gs0': minimum conductance (mol m⁻² s⁻¹),
                'g1': marginal water use efficiency (√kPa)
            }

        Returns:
            Predicted stomatal conductance (mol m⁻² s⁻¹)
        """
        A = np.asarray(data['A'])
        VPD = np.asarray(data['VPD'])
        Ca = data.get('Ca', 400.0)
        if not isinstance(Ca, (int, float)):
            Ca = np.asarray(Ca)

        gs0 = parameters['gs0']
        g1 = parameters['g1']

        # Only use positive A values (stomata close when A < 0)
        A_pos = np.maximum(A, 0.0)

        # BTA model (unified stomatal optimization)
        # Avoid division by zero
        sqrt_VPD = np.sqrt(np.maximum(VPD, 0.01))

        gs = gs0 + g1 * A_pos / (Ca * sqrt_VPD)

        return gs

    def parameter_info(self) -> dict:
        return {
            'gs0': {
                'default': 0.01,
                'bounds': (0.0, 0.1),
                'units': 'mol m⁻² s⁻¹',
                'description': 'Minimum stomatal conductance',
                'symbol': 'g_{s0}'
            },
            'g1': {
                'default': 4.0,
                'bounds': (0.5, 15.0),
                'units': '√kPa',
                'description': 'Marginal water use efficiency',
                'symbol': 'g_1'
            }
        }

    def required_data(self) -> list:
        return ['A', 'VPD', 'gs']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data."""
        gs = np.asarray(data['gs'])

        # gs0: minimum observed conductance
        gs0_guess = np.min(gs[gs > 0]) if np.any(gs > 0) else 0.01
        gs0_guess = min(gs0_guess, 0.05)

        # g1: use typical value
        g1_guess = 4.0

        return {
            'gs0': gs0_guess,
            'g1': g1_guess
        }

    def plot(self, data: dict, parameters: dict, show: bool = True, save: str = None):
        """Plot stomatal conductance fit results.

        Args:
            data: Data dictionary with 'A', 'VPD', and 'gs'
            parameters: Fitted parameters
            show: Whether to display the plot (default: True)
            save: Filename to save plot (default: None)
        """
        import matplotlib.pyplot as plt

        # Extract parameters
        gs0 = parameters['gs0']
        g1 = parameters['g1']

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

        # Right plot: gs vs A
        A = np.asarray(data['A'])

        ax2.scatter(A, observed, s=100, alpha=0.6, color='black',
                   edgecolors='black', linewidth=0.5, label='Observed', zorder=3)

        # Plot fitted curve
        # Sort by A for smooth line
        sort_idx = np.argsort(A)
        ax2.plot(A[sort_idx], predicted[sort_idx], 'r-', linewidth=2.5,
                label='Model fit', zorder=2)

        ax2.set_xlabel('Net Assimilation, A (μmol m⁻² s⁻¹)', fontsize=12)
        ax2.set_ylabel('Stomatal Conductance, gs (mol m⁻² s⁻¹)', fontsize=12)
        ax2.set_title('Stomatal Conductance Response', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)

        # Add parameter info box
        param_text = f"Fitted Parameters:\n"
        param_text += f"gs0 = {gs0:.4f} mol m⁻² s⁻¹\n"
        param_text += f"g1 = {g1:.3f} √kPa\n\n"
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
