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
