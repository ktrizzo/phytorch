"""Base class for all PhyTorch models."""

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class Model(ABC):
    """Base class for all PhyTorch physiological models.

    All models must implement three abstract methods:
    - forward(): Compute model predictions from data and parameters
    - parameter_info(): Define parameter metadata (defaults, bounds, units)
    - required_data(): Specify required input data fields

    Models can optionally override:
    - initial_guess(): Provide smarter parameter initialization from data
    """

    @abstractmethod
    def forward(self, data: Dict, parameters: Dict) -> np.ndarray:
        """Compute model predictions.

        Args:
            data: Input data dict (e.g., {'psi': array, 'T': array})
                  Keys must match those returned by required_data()
            parameters: Model parameters dict (e.g., {'Kmax': 10.0, 'psi50': -1.5})
                       Keys must match those in parameter_info()

        Returns:
            Model predictions as numpy array matching length of data inputs
        """
        pass

    @abstractmethod
    def parameter_info(self) -> Dict:
        """Return parameter metadata.

        Returns:
            Dict with structure:
            {
                'param_name': {
                    'default': float,      # Default initial value
                    'bounds': (low, high), # Parameter bounds (inclusive)
                    'units': str,          # Physical units
                    'description': str,    # Brief description
                    'symbol': str          # LaTeX/Unicode symbol (optional)
                }
            }

        Example:
            {
                'Kmax': {
                    'default': 10.0,
                    'bounds': (0.0, np.inf),
                    'units': 'mmol m⁻² s⁻¹ MPa⁻¹',
                    'description': 'Maximum hydraulic conductance',
                    'symbol': 'K_max'
                }
            }
        """
        pass

    @abstractmethod
    def required_data(self) -> List[str]:
        """Return list of required data fields.

        Returns:
            List of required keys that must be present in data dict

        Example:
            ['psi', 'K']  # Requires water potential and conductance data
        """
        pass

    def initial_guess(self, data: Dict) -> Dict:
        """Estimate initial parameter values from data.

        Default implementation returns 'default' value from parameter_info().
        Override this method to provide smarter, data-driven initialization.

        Args:
            data: Input data dict

        Returns:
            Dict of initial parameter guesses

        Example:
            def initial_guess(self, data):
                K = np.asarray(data['K'])
                return {
                    'Kmax': np.max(K) * 1.1,  # Slightly above max observed
                    'psi50': -1.5,  # Typical value
                    's': 2.0
                }
        """
        return {name: info['default']
                for name, info in self.parameter_info().items()}

    def validate_data(self, data: Dict) -> None:
        """Validate that data contains all required fields.

        Args:
            data: Input data dict

        Raises:
            ValueError: If required data fields are missing
        """
        required = self.required_data()
        missing = [field for field in required if field not in data]
        if missing:
            raise ValueError(
                f"Missing required data fields: {missing}. "
                f"Model requires: {required}"
            )

    def validate_parameters(self, parameters: Dict) -> None:
        """Validate that all required parameters are present.

        Args:
            parameters: Model parameters dict

        Raises:
            ValueError: If required parameters are missing
        """
        param_info = self.parameter_info()
        required = set(param_info.keys())
        provided = set(parameters.keys())
        missing = required - provided
        if missing:
            raise ValueError(
                f"Missing required parameters: {missing}. "
                f"Model requires: {required}"
            )

    def __repr__(self):
        return f"{self.__class__.__name__}()"
