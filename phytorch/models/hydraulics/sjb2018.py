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
