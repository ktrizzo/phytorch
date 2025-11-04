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
    CANONICAL_TYPES = {
        'planophile': {'mu': 2.770, 'nu': 1.172},
        'erectophile': {'mu': 1.172, 'nu': 2.770},
        'plagiophile': {'mu': 3.326, 'nu': 3.326},
        'extremophile': {'mu': 0.433, 'nu': 0.433},
        'uniform': {'mu': 1.000, 'nu': 1.000},
        'spherical': {'mu': 1.101, 'nu': 1.930}
    }

    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute probability density of leaf angles.

        Args:
            data: {'theta': leaf inclination angle from horizontal (degrees, 0-90)}
            parameters: {
                'mu': first shape parameter,
                'nu': second shape parameter
            }

        Returns:
            Predicted probability density
        """
        theta = np.asarray(data['theta'])
        mu = parameters['mu']
        nu = parameters['nu']

        # Convert to radians
        theta_rad = np.deg2rad(theta)

        # Beta distribution for leaf angles
        # f(θ) = sin^(μ-1)(θ) · cos^(ν-1)(θ) / (B(μ,ν) · 90)
        sin_term = np.power(np.sin(theta_rad), mu - 1)
        cos_term = np.power(np.cos(theta_rad), nu - 1)
        normalization = beta_function(mu, nu) * 90.0

        return (sin_term * cos_term) / normalization

    def parameter_info(self) -> dict:
        return {
            'mu': {
                'default': 1.5,
                'bounds': (0.1, 5.0),
                'units': '',
                'description': 'First shape parameter (controls horizontal tendency)',
                'symbol': 'μ'
            },
            'nu': {
                'default': 1.5,
                'bounds': (0.1, 5.0),
                'units': '',
                'description': 'Second shape parameter (controls vertical tendency)',
                'symbol': 'ν'
            }
        }

    def required_data(self) -> list:
        return ['theta', 'frequency']

    def initial_guess(self, data: dict) -> dict:
        """Estimate initial parameters from data.

        Uses mean angle to estimate shape parameters:
        - Low mean angle → planophile (high mu, low nu)
        - High mean angle → erectophile (low mu, high nu)
        - Mid-range → spherical or plagiophile
        """
        theta = np.asarray(data['theta'])
        frequency = np.asarray(data['frequency']) if 'frequency' in data else np.ones_like(theta)

        # Compute weighted mean angle
        mean_angle = np.average(theta, weights=frequency)

        # Estimate parameters based on mean angle
        if mean_angle < 30:
            # Planophile-like
            mu_guess, nu_guess = 2.5, 1.2
        elif mean_angle > 60:
            # Erectophile-like
            mu_guess, nu_guess = 1.2, 2.5
        elif 40 <= mean_angle <= 50:
            # Plagiophile-like
            mu_guess, nu_guess = 3.0, 3.0
        else:
            # Spherical-like
            mu_guess, nu_guess = 1.1, 1.9

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
