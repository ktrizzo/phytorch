"""Leaf angle distribution model for canopy architecture."""

import numpy as np
from scipy.special import beta as beta_function
from phytorch.models.base import Model


class LeafAngleDistribution(Model):
    """Beta distribution model for leaf inclination angle distribution.

    Model equation:
        f(θ) = (sin^(μ-1)(θ) · cos^(ν-1)(θ)) / (B(μ, ν) · 90)

    where θ is leaf inclination angle from horizontal (0-90 degrees),
    μ and ν are shape parameters, and B is the beta function.

    The fitted distribution is classified into one of six canonical types
    defined by de Wit (1965):
    - Planophile: mostly horizontal leaves (μ=2.770, ν=1.172)
    - Erectophile: mostly vertical leaves (μ=1.172, ν=2.770)
    - Plagiophile: mostly oblique leaves (μ=3.326, ν=3.326)
    - Extremophile: both horizontal and vertical (μ=0.433, ν=0.433)
    - Uniform: equal distribution (μ=1.000, ν=1.000)
    - Spherical: spherical distribution (μ=1.101, ν=1.930)

    Reference:
        de Wit, C.T. (1965). Photosynthesis of Leaf Canopies.
        Agricultural Research Reports No. 663, Pudoc, Wageningen.
    """

    # Canonical distribution parameters from de Wit (1965)
    # Note: Swapped mu/nu from literature because angle is from horizontal not zenith
    CANONICAL_TYPES = {
        'planophile': {'mu': 1.172, 'nu': 2.770},  # Horizontal leaves
        'erectophile': {'mu': 2.770, 'nu': 1.172},  # Vertical leaves
        'plagiophile': {'mu': 3.326, 'nu': 3.326},  # Oblique leaves
        'extremophile': {'mu': 0.433, 'nu': 0.433},  # Bimodal
        'uniform': {'mu': 1.000, 'nu': 1.000},  # Uniform
        'spherical': {'mu': 1.930, 'nu': 1.101}  # Spherical
    }

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute probability density of leaf angles.

        Args:
            data: {'x': leaf inclination angle from horizontal (degrees, 0-90)}
            parameters: {
                'mu': first shape parameter,
                'nu': second shape parameter
            }

        Returns:
            Predicted probability density (per degree)
        """
        theta = np.asarray(data['x'])
        mu = parameters['mu']
        nu = parameters['nu']

        # Transform theta from [0, 90] degrees to t in [0, 1]
        # t = theta / 90
        t = theta / 90.0
        t = np.clip(t, 1e-10, 1.0 - 1e-10)  # Avoid boundary issues

        # Beta distribution: f(t) = t^(μ-1) · (1-t)^(ν-1) / B(μ,ν)
        # Density per degree: f_theta(θ) = f(t) / 90
        beta_norm = beta_function(mu, nu)
        t_term = np.power(t, mu - 1)
        one_minus_t_term = np.power(1.0 - t, nu - 1)

        # Probability density per degree
        return (t_term * one_minus_t_term) / (beta_norm * 90.0)

    def parameter_info(self) -> dict:
        return {
            'mu': {
                'default': 1.5,
                'bounds': (0.3, 5.0),
                'units': '',
                'description': 'First shape parameter (controls horizontal tendency)',
                'symbol': 'μ'
            },
            'nu': {
                'default': 1.5,
                'bounds': (0.3, 5.0),
                'units': '',
                'description': 'Second shape parameter (controls vertical tendency)',
                'symbol': 'ν'
            }
        }

    def required_data(self) -> list:
        return ['x', 'y']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data.

        Uses mean angle to estimate shape parameters:
        - Low mean angle → planophile (high mu, low nu)
        - High mean angle → erectophile (low mu, high nu)
        - Mid-range → spherical or plagiophile
        """
        theta = np.asarray(data['x'])
        frequency = np.asarray(data['y'])

        # Compute weighted mean angle
        mean_angle = np.average(theta, weights=frequency)

        # Estimate parameters based on mean angle
        # Use canonical values as starting points (adjusted for angle from horizontal)
        if mean_angle < 30:
            # Planophile-like (mostly horizontal)
            mu_guess, nu_guess = 1.172, 2.770
        elif mean_angle > 60:
            # Erectophile-like (mostly vertical)
            mu_guess, nu_guess = 2.770, 1.172
        elif 40 <= mean_angle <= 50:
            # Plagiophile-like (mostly oblique)
            mu_guess, nu_guess = 3.326, 3.326
        else:
            # Spherical-like
            mu_guess, nu_guess = 1.930, 1.101

        return {
            'mu': mu_guess,
            'nu': nu_guess
        }

    def classify(self, parameters: dict) -> dict:
        """Classify fitted distribution into canonical type.

        Args:
            parameters: Fitted parameters {'mu': value, 'nu': value}

        Returns:
            Dictionary with:
                'type': name of closest canonical type
                'distance': Euclidean distance to canonical type
                'mu': fitted mu parameter
                'nu': fitted nu parameter
                'canonical_mu': mu of canonical type
                'canonical_nu': nu of canonical type
        """
        mu = parameters['mu']
        nu = parameters['nu']

        # Find closest canonical type by Euclidean distance
        min_distance = float('inf')
        best_type = None

        for type_name, params in self.CANONICAL_TYPES.items():
            distance = np.sqrt((mu - params['mu'])**2 + (nu - params['nu'])**2)
            if distance < min_distance:
                min_distance = distance
                best_type = type_name

        return {
            'type': best_type,
            'distance': min_distance,
            'mu': mu,
            'nu': nu,
            'canonical_mu': self.CANONICAL_TYPES[best_type]['mu'],
            'canonical_nu': self.CANONICAL_TYPES[best_type]['nu']
        }

    def plot(self, data: dict, parameters: dict, show: bool = True, save: str = None):
        """Custom plot for leaf angle distribution showing canopy classification.

        Args:
            data: Data dictionary with 'x' (angles) and 'y' (frequencies)
            parameters: Fitted parameters
            show: Whether to display the plot (default: True)
            save: Filename to save plot (default: None)
        """
        import matplotlib.pyplot as plt

        # Classify canopy type
        classification = self.classify(parameters)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data
        ax.scatter(data['x'], data['y'], s=100, alpha=0.6, color='black',
                   edgecolors='black', linewidth=0.5, label='Measured', zorder=3)

        # Plot fitted curve
        theta_fine = np.linspace(0, 90, 200)
        predicted = self.forward({'x': theta_fine}, parameters)
        ax.plot(theta_fine, predicted, 'r-', linewidth=2.5,
                label=f'Fit: {classification["type"]} (μ={parameters["mu"]:.2f}, ν={parameters["nu"]:.2f})',
                zorder=2)

        # Labels and title
        ax.set_xlabel('Leaf Angle from Horizontal (degrees)', fontsize=12)
        ax.set_ylabel('Frequency (probability density)', fontsize=12)
        ax.set_title(f'Leaf Angle Distribution - {classification["type"].upper()}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 90)
        ax.set_ylim(bottom=0)

        # Add classification info box
        info_text = f"Canonical: μ={classification['canonical_mu']:.2f}, ν={classification['canonical_nu']:.2f}\n"
        info_text += f"Distance: {classification['distance']:.3f}"
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
                fontsize=9, va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=200, bbox_inches='tight')

        if show:
            plt.show()

        return fig, ax
