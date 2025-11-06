"""
Farquhar-von Caemmerer-Berry (FvCB) photosynthesis model for PhyTorch.

This module wraps the legacy PyTorch FvCB implementation to work with
the unified fit(model, data, options) API while preserving all functionality.

Usage:
    from phytorch import fit
    from phytorch.models.photosynthesis import FvCB

    # Prepare A-Ci curve data
    data = {
        'A': A_values,      # Net photosynthesis (μmol m⁻² s⁻¹)
        'Ci': Ci_values,    # Intercellular CO₂ (ppm)
        'Qin': Q_values,    # PPFD (μmol m⁻² s⁻¹)
        'Tleaf': T_values,  # Leaf temperature (°C)
        'CurveID': IDs      # Curve identifiers
    }

    # Fit model
    model = FvCB(light_response=1, temp_response=1)
    result = fit(model, data)

    # Access results
    print(result.parameters)
    result.plot()
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Optional

from phytorch.models.base import Model
from .fvcb_core_legacy import (
    allparameters,
    LightResponse,
    TemperatureResponse,
    FvCB as FvCBCore,
    Loss
)
from .fvcb_data_legacy import initLicordata


class FvCB(Model, nn.Module):
    """
    Farquhar-von Caemmerer-Berry C3 photosynthesis model.

    Combines Rubisco-limited (Ac), RuBP-regeneration-limited (Aj), and
    TPU-limited (Ap) photosynthesis rates with temperature and light responses.

    Args:
        light_response: Light response type (0=none, 1=fit alpha, 2=fit alpha+theta)
        temp_response: Temperature response type (0=none, 1=fit dHa, 2=fit dHa+Topt)
        fit_gm: Fit mesophyll conductance (default: False)
        fit_gamma: Fit CO₂ compensation point (default: False)
        fit_Kc: Fit Michaelis constant for CO₂ (default: False)
        fit_Ko: Fit Michaelis constant for O₂ (default: False)
        fit_Rd: Fit dark respiration (default: True)
        preprocess: Preprocess A-Ci curves (smoothing, outlier removal) (default: True)
        lightresp_id: List of CurveIDs that are light response curves (default: None)
        verbose: Print model configuration (default: True)
    """

    # Flag for torch optimizer
    use_torch_optimizer = True

    def __init__(
        self,
        light_response: int = 1,
        temp_response: int = 1,
        fit_gm: bool = False,
        fit_gamma: bool = False,
        fit_Kc: bool = False,
        fit_Ko: bool = False,
        fit_Rd: bool = True,
        preprocess: bool = True,
        lightresp_id: Optional[list] = None,
        verbose: bool = True
    ):
        Model.__init__(self)
        nn.Module.__init__(self)

        self.light_response_type = light_response
        self.temp_response_type = temp_response
        self.fit_gm = fit_gm
        self.fit_gamma = fit_gamma
        self.fit_Kc = fit_Kc
        self.fit_Ko = fit_Ko
        self.fit_Rd = fit_Rd
        self.preprocess = preprocess
        self.lightresp_id = lightresp_id
        self.verbose = verbose

        # Will be initialized in _prepare_data
        self.lcd = None
        self.core_model = None
        self.loss_fn = None
        self._data_cache = None

    def _prepare_data(self, data: Dict):
        """Convert dict data to initLicordata format."""
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data

        # Ensure required columns exist
        required = ['A', 'Ci', 'Qin', 'Tleaf']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add CurveID if not present
        if 'CurveID' not in df.columns:
            df['CurveID'] = 0

        # Initialize Licor data structure
        self.lcd = initLicordata(
            df,
            preprocess=self.preprocess,
            lightresp_id=self.lightresp_id,
            printout=self.verbose
        )

        # Create core FvCB model
        self.core_model = FvCBCore(
            self.lcd,
            LightResp_type=self.light_response_type,
            TempResp_type=self.temp_response_type,
            fitgm=self.fit_gm,
            fitgamma=self.fit_gamma,
            fitKc=self.fit_Kc,
            fitKo=self.fit_Ko,
            fitRd=self.fit_Rd,
            printout=self.verbose
        )

        # Create loss function
        self.loss_fn = Loss(
            self.lcd,
            fitApCi=500,
            fitCorrelation=True,
            weakconstiter=10000
        )

        # Cache for reuse
        self._data_cache = data

    def forward(self, data: Optional[Dict] = None, parameters: Optional[Dict] = None):
        """
        Compute photosynthesis predictions.

        Args:
            data: Input data dict (uses cached if None)
            parameters: Not used (model uses nn.Parameters internally)

        Returns:
            A: Net photosynthesis predictions (or tuple of (A, Ac, Aj, Ap) internally)
        """
        # Use cached data if available
        if data is None:
            if self._data_cache is None:
                raise ValueError("No data available. Call with data first.")
            data = self._data_cache
        elif self.lcd is None or data is not self._data_cache:
            self._prepare_data(data)

        # Forward through core model
        A, Ac, Aj, Ap = self.core_model()

        # Return tuple for internal use, array for API
        if parameters is None:
            return (A, Ac, Aj, Ap)
        else:
            return A.detach().cpu().numpy()

    def compute_loss(self, data: Dict):
        """
        Compute loss for torch optimizer.

        Args:
            data: Input data dict

        Returns:
            loss: Scalar loss tensor
        """
        # Prepare data if needed
        if self.lcd is None or data is not self._data_cache:
            self._prepare_data(data)

        # Forward pass
        A, Ac, Aj, Ap = self.core_model()

        # Compute loss with current iteration (use 0 for simplicity)
        loss = self.loss_fn(self.core_model, A, Ac, Aj, Ap, iter=0)

        return loss

    def parameter_info(self) -> Dict:
        """Return parameter metadata."""
        # Basic parameters - actual bounds depend on configuration
        params = {
            'Vcmax25': {
                'default': 100.0,
                'bounds': (20.0, 300.0),
                'units': 'μmol m⁻² s⁻¹',
                'description': 'Maximum Rubisco carboxylation rate at 25°C'
            },
            'Jmax25': {
                'default': 200.0,
                'bounds': (40.0, 600.0),
                'units': 'μmol m⁻² s⁻¹',
                'description': 'Maximum electron transport rate at 25°C'
            },
            'TPU25': {
                'default': 25.0,
                'bounds': (5.0, 100.0),
                'units': 'μmol m⁻² s⁻¹',
                'description': 'Triose phosphate utilization rate at 25°C'
            },
            'Rd25': {
                'default': 1.5,
                'bounds': (0.0, 10.0),
                'units': 'μmol m⁻² s⁻¹',
                'description': 'Dark respiration rate at 25°C'
            }
        }

        return params

    def required_data(self) -> list:
        """Return required data fields."""
        return ['Ci', 'Qin', 'Tleaf', 'A']

    def initial_guess(self, data: Dict) -> Dict:
        """Estimate initial parameters from data."""
        # Use defaults - the model initializes with reasonable values
        return {
            'Vcmax25': 100.0,
            'Jmax25': 200.0,
            'TPU25': 25.0,
            'Rd25': 1.5
        }

    def get_observed_data(self) -> np.ndarray:
        """Return observed data for R² calculation (after preprocessing)."""
        if self.lcd is None:
            raise ValueError("Model not initialized with data yet")
        return self.lcd.A.cpu().numpy()
