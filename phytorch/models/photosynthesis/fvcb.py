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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        """Return parameter metadata for all possible model parameters."""
        params = {
            # Main biochemical parameters
            'Vcmax25': {
                'default': 100.0,
                'bounds': (20.0, 300.0),
                'units': 'umol m-2 s-1',
                'description': 'Maximum Rubisco carboxylation rate at 25C',
                'symbol': 'Vcmax25'
            },
            'Jmax25': {
                'default': 200.0,
                'bounds': (40.0, 600.0),
                'units': 'umol m-2 s-1',
                'description': 'Maximum electron transport rate at 25C',
                'symbol': 'Jmax25'
            },
            'TPU25': {
                'default': 25.0,
                'bounds': (5.0, 100.0),
                'units': 'umol m-2 s-1',
                'description': 'Triose phosphate utilization rate at 25C',
                'symbol': 'TPU25'
            },
            'Rd25': {
                'default': 1.5,
                'bounds': (0.0, 10.0),
                'units': 'umol m-2 s-1',
                'description': 'Dark respiration rate at 25C',
                'symbol': 'Rd25'
            },

            # Light response parameters
            'alpha': {
                'default': 0.9,
                'bounds': (0.0, 1.0),
                'units': 'mol e- / mol photon',
                'description': 'Quantum yield of electron transport',
                'symbol': 'alpha'
            },
            'theta': {
                'default': 0.7,
                'bounds': (0.0, 1.0),
                'units': 'dimensionless',
                'description': 'Curvature factor for light response',
                'symbol': 'theta'
            },

            # Temperature response parameters - Activation energies
            'Vcmax_dHa': {
                'default': 73.0,
                'bounds': (50.0, 120.0),
                'units': 'kJ mol-1',
                'description': 'Activation energy for Vcmax',
                'symbol': 'dHa_Vcmax'
            },
            'Jmax_dHa': {
                'default': 33.0,
                'bounds': (20.0, 80.0),
                'units': 'kJ mol-1',
                'description': 'Activation energy for Jmax',
                'symbol': 'dHa_Jmax'
            },
            'TPU_dHa': {
                'default': 73.0,
                'bounds': (50.0, 120.0),
                'units': 'kJ mol-1',
                'description': 'Activation energy for TPU',
                'symbol': 'dHa_TPU'
            },

            # Temperature response parameters - Optimal temperatures (peaked Arrhenius)
            'Vcmax_Topt': {
                'default': 311.15,
                'bounds': (298.15, 323.15),
                'units': 'K',
                'description': 'Optimal temperature for Vcmax',
                'symbol': 'Topt_Vcmax'
            },
            'Jmax_Topt': {
                'default': 311.15,
                'bounds': (298.15, 323.15),
                'units': 'K',
                'description': 'Optimal temperature for Jmax',
                'symbol': 'Topt_Jmax'
            },
            'TPU_Topt': {
                'default': 311.15,
                'bounds': (298.15, 323.15),
                'units': 'K',
                'description': 'Optimal temperature for TPU',
                'symbol': 'Topt_TPU'
            },

            # Optional biochemical parameters
            'gm': {
                'default': 0.4,
                'bounds': (0.01, 2.0),
                'units': 'mol m-2 s-1 bar-1',
                'description': 'Mesophyll conductance to CO2',
                'symbol': 'gm'
            },
            'Gamma25': {
                'default': 42.75,
                'bounds': (30.0, 60.0),
                'units': 'umol mol-1',
                'description': 'CO2 compensation point at 25C',
                'symbol': 'Gamma25'
            },
            'Kc25': {
                'default': 404.9,
                'bounds': (200.0, 800.0),
                'units': 'umol mol-1',
                'description': 'Michaelis constant for CO2 at 25C',
                'symbol': 'Kc25'
            },
            'Ko25': {
                'default': 278.4,
                'bounds': (100.0, 500.0),
                'units': 'mmol mol-1',
                'description': 'Michaelis constant for O2 at 25C',
                'symbol': 'Ko25'
            },
            'alpha_G': {
                'default': 0.5,
                'bounds': (0.0, 1.0),
                'units': 'dimensionless',
                'description': 'Stoichiometric ratio of orthophosphate (Pi) consumption in oxygenation',
                'symbol': 'alpha_G'
            },
            'Rd_dHa': {
                'default': 46.39,
                'bounds': (20.0, 80.0),
                'units': 'kJ mol-1',
                'description': 'Activation energy for Rd',
                'symbol': 'dHa_Rd'
            },
            'Gamma_dHa': {
                'default': 37.83,
                'bounds': (20.0, 60.0),
                'units': 'kJ mol-1',
                'description': 'Activation energy for Gamma',
                'symbol': 'dHa_Gamma'
            },
            'Kc_dHa': {
                'default': 79.43,
                'bounds': (50.0, 120.0),
                'units': 'kJ mol-1',
                'description': 'Activation energy for Kc',
                'symbol': 'dHa_Kc'
            },
            'Ko_dHa': {
                'default': 36.38,
                'bounds': (20.0, 60.0),
                'units': 'kJ mol-1',
                'description': 'Activation energy for Ko',
                'symbol': 'dHa_Ko'
            },
            'Vcmax_dHd': {
                'default': 200.0,
                'bounds': (150.0, 250.0),
                'units': 'kJ mol-1',
                'description': 'Deactivation energy for Vcmax',
                'symbol': 'dHd_Vcmax'
            },
            'Jmax_dHd': {
                'default': 200.0,
                'bounds': (150.0, 250.0),
                'units': 'kJ mol-1',
                'description': 'Deactivation energy for Jmax',
                'symbol': 'dHd_Jmax'
            },
            'TPU_dHd': {
                'default': 201.8,
                'bounds': (150.0, 250.0),
                'units': 'kJ mol-1',
                'description': 'Deactivation energy for TPU',
                'symbol': 'dHd_TPU'
            },
            'O': {
                'default': 213.5,
                'bounds': (200.0, 230.0),
                'units': 'mmol mol-1',
                'description': 'Oxygen concentration',
                'symbol': 'O'
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

    def get_preprocessed_data(self) -> Dict:
        """Return preprocessed data dict (after filtering/outlier removal)."""
        if self.lcd is None:
            raise ValueError("Model not initialized with data yet")

        result = {
            'A': self.lcd.A.cpu().numpy(),
            'Ci': self.lcd.Ci.cpu().numpy(),
            'Qin': self.lcd.Q.cpu().numpy(),
            'Tleaf': self.lcd.Tleaf.cpu().numpy(),
        }

        # Handle CurveID - may be tensor or numpy array, only include if valid
        if hasattr(self.lcd, 'FGs_idx') and self.lcd.FGs_idx is not None:
            curve_id = self.lcd.FGs_idx
            if hasattr(curve_id, 'cpu'):
                curve_id = curve_id.cpu().numpy()
            if len(curve_id) == len(result['A']):
                result['CurveID'] = curve_id

        return result

    def get_all_parameters(self) -> Dict:
        """
        Return all model parameters, including both fitted and default values.

        This extracts:
        - Fitted parameters from nn.Parameter objects
        - Default values for parameters that weren't fitted
        - All biochemical constants and temperature dependencies

        Returns:
            Dictionary of all parameter names and values (flat structure)
        """
        if self.core_model is None:
            raise ValueError("Model not initialized. Call fit() first.")

        all_params = {}

        # Get default parameters object
        from .fvcb_core_legacy import allparameters
        ap = allparameters()

        # Extract all nn.Parameters from the model (these are the fitted ones)
        for name, param in self.core_model.named_parameters():
            if param.numel() > 1:
                # Take mean for array parameters (across curves)
                all_params[name] = param.detach().cpu().mean().item()
            else:
                all_params[name] = param.detach().cpu().item()

        # Main biochemical parameters (Vcmax25, Jmax25, TPU25, Rd25 should already be in all_params from nn.Parameters)

        # Light response parameters
        if 'LightResponse.alpha' not in all_params:
            all_params['alpha'] = ap.alpha.item()
        else:
            all_params['alpha'] = all_params.pop('LightResponse.alpha')

        if 'LightResponse.theta' not in all_params:
            all_params['theta'] = ap.theta.item()
        else:
            all_params['theta'] = all_params.pop('LightResponse.theta')

        # Temperature response - activation energies
        if 'TempResponse.dHa_Vcmax' not in all_params:
            all_params['Vcmax_dHa'] = ap.dHa_Vcmax.item()
        else:
            all_params['Vcmax_dHa'] = all_params.pop('TempResponse.dHa_Vcmax')

        if 'TempResponse.dHa_Jmax' not in all_params:
            all_params['Jmax_dHa'] = ap.dHa_Jmax.item()
        else:
            all_params['Jmax_dHa'] = all_params.pop('TempResponse.dHa_Jmax')

        if 'TempResponse.dHa_TPU' not in all_params:
            all_params['TPU_dHa'] = ap.dHa_TPU.item()
        else:
            all_params['TPU_dHa'] = all_params.pop('TempResponse.dHa_TPU')

        # Temperature response - optimal temperatures (peaked Arrhenius)
        if 'TempResponse.Topt_Vcmax' not in all_params:
            all_params['Vcmax_Topt'] = ap.Topt_Vcmax.item()
        else:
            all_params['Vcmax_Topt'] = all_params.pop('TempResponse.Topt_Vcmax')

        if 'TempResponse.Topt_Jmax' not in all_params:
            all_params['Jmax_Topt'] = ap.Topt_Jmax.item()
        else:
            all_params['Jmax_Topt'] = all_params.pop('TempResponse.Topt_Jmax')

        if 'TempResponse.Topt_TPU' not in all_params:
            all_params['TPU_Topt'] = ap.Topt_TPU.item()
        else:
            all_params['TPU_Topt'] = all_params.pop('TempResponse.Topt_TPU')

        # Temperature response - deactivation energies
        all_params['Vcmax_dHd'] = ap.dHd_Vcmax.item()
        all_params['Jmax_dHd'] = ap.dHd_Jmax.item()
        all_params['TPU_dHd'] = ap.dHd_TPU.item()

        # Rd temperature response
        all_params['Rd_dHa'] = ap.dHa_Rd.item()

        # CO2 compensation point and its temperature response
        if self.fit_gamma and hasattr(self.core_model, 'Gamma25'):
            all_params['Gamma25'] = self.core_model.Gamma25.detach().cpu().mean().item()
        else:
            all_params['Gamma25'] = ap.Gamma25.item()
        all_params['Gamma_dHa'] = ap.dHa_Gamma.item()

        # Michaelis constants for CO2 and O2 and their temperature responses
        if self.fit_Kc and hasattr(self.core_model, 'Kc25'):
            all_params['Kc25'] = self.core_model.Kc25.detach().cpu().mean().item()
        else:
            all_params['Kc25'] = ap.Kc25.item()
        all_params['Kc_dHa'] = ap.dHa_Kc.item()

        if self.fit_Ko and hasattr(self.core_model, 'Ko25'):
            all_params['Ko25'] = self.core_model.Ko25.detach().cpu().mean().item()
        else:
            all_params['Ko25'] = ap.Ko25.item()
        all_params['Ko_dHa'] = ap.dHa_Ko.item()

        # Oxygen concentration
        all_params['O'] = ap.oxy.item()

        # Mesophyll conductance
        if self.fit_gm and hasattr(self.core_model, 'gm'):
            all_params['gm'] = self.core_model.gm.detach().cpu().mean().item()
        else:
            all_params['gm'] = ap.gm.item()

        # Alpha_G (stoichiometric ratio)
        if hasattr(self.core_model, 'fitag') and self.core_model.fitag and hasattr(self.core_model, '_FvCB__alphaG_r'):
            all_params['alpha_G'] = self.core_model._FvCB__alphaG_r.detach().cpu().mean().item()
        else:
            all_params['alpha_G'] = ap.alphaG_r.item()

        return all_params

    def plot(self, data: Optional[Dict] = None, parameters: Optional[Dict] = None,
             show: bool = True, save: str = None):
        """
        Generate comprehensive 6-panel diagnostic plot for FvCB model fit.

        Args:
            data: Input data dict (uses cached if None)
            parameters: Not used (model uses internal parameters)
            show: Display plot (default: True)
            save: Save to file path (default: None)
        """
        # Use cached data if available
        if data is None:
            if self._data_cache is None:
                raise ValueError("No data available. Call fit() first.")
            data = self._data_cache
        elif self.lcd is None or data is not self._data_cache:
            self._prepare_data(data)

        # Get predictions and observed data
        with torch.no_grad():
            A_pred, Ac, Aj, Ap = self.core_model()
            A_pred = A_pred.cpu().numpy()

        A_obs = self.lcd.A.cpu().numpy()
        Ci_obs = self.lcd.Ci.cpu().numpy()
        Q_obs = self.lcd.Q.cpu().numpy()
        Tleaf_obs = self.lcd.Tleaf.cpu().numpy()

        # Calculate R²
        ss_res = np.sum((A_obs - A_pred) ** 2)
        ss_tot = np.sum((A_obs - np.mean(A_obs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Get mean values for plotting grids
        T_plot_K = 298.15  # 25°C for response curves
        T_plot_C = 25.0
        Ci_mean = Ci_obs.mean()
        Q_mean = 2000.0

        # Helper function to evaluate model on grid using FvCB equations with fitted parameters
        def evaluate_model_grid(Ci_grid, Q_grid, T_grid):
            """
            Evaluate model on a grid using FvCB equations with the fitted parameters.
            Uses the complete parameter set to ensure accurate predictions.
            """
            # Flatten inputs
            Ci_flat = Ci_grid.flatten()
            Q_flat = Q_grid.flatten()
            T_flat = T_grid.flatten()

            # Get all fitted/fixed parameters
            params = self.get_all_parameters()

            # Constants
            R = 0.008314  # kJ/(mol·K)
            Tref = 298.15  # K

            # Compute temperature-scaled parameters for each point
            A_result = []
            for Ci_val, Q_val, T_val in zip(Ci_flat, Q_flat, T_flat):
                # Simple Arrhenius for Vcmax
                Vcmax_arr = params['Vcmax25'] * np.exp(params['Vcmax_dHa'] / R * (1/Tref - 1/T_val))

                # Apply peaked Arrhenius deactivation
                dHd_dHa = params['Vcmax_dHd'] / params['Vcmax_dHa']
                dHd_dHa = max(dHd_dHa, 1.0001)
                log_term = np.log(dHd_dHa - 1)
                num = 1 + np.exp(params['Vcmax_dHd'] / R * (1/params['Vcmax_Topt'] - 1/Tref) - log_term)
                den = 1 + np.exp(params['Vcmax_dHd'] / R * (1/params['Vcmax_Topt'] - 1/T_val) - log_term)
                Vcmax = Vcmax_arr * num / den

                # Same for Jmax
                Jmax_arr = params['Jmax25'] * np.exp(params['Jmax_dHa'] / R * (1/Tref - 1/T_val))
                dHd_dHa = params['Jmax_dHd'] / params['Jmax_dHa']
                dHd_dHa = max(dHd_dHa, 1.0001)
                log_term = np.log(dHd_dHa - 1)
                num = 1 + np.exp(params['Jmax_dHd'] / R * (1/params['Jmax_Topt'] - 1/Tref) - log_term)
                den = 1 + np.exp(params['Jmax_dHd'] / R * (1/params['Jmax_Topt'] - 1/T_val) - log_term)
                Jmax = Jmax_arr * num / den

                # Rd with simple Arrhenius
                Rd = params['Rd25'] * np.exp(params['Rd_dHa'] / R * (1/Tref - 1/T_val))

                # Michaelis constants
                Kc = params['Kc25'] * np.exp(params['Kc_dHa'] / R * (1/Tref - 1/T_val))
                Ko = params['Ko25'] * np.exp(params['Ko_dHa'] / R * (1/Tref - 1/T_val))
                Gamma = params['Gamma25'] * np.exp(params['Gamma_dHa'] / R * (1/Tref - 1/T_val))

                # Electron transport rate
                alpha = params['alpha']
                theta = params['theta']
                J = (alpha * Q_val + Jmax - np.sqrt((alpha * Q_val + Jmax)**2 - 4 * theta * alpha * Q_val * Jmax)) / (2 * theta)

                # Rubisco-limited rate
                Kco = Kc * (1 + params['O'] / Ko)
                Ac = Vcmax * (Ci_val - Gamma) / (Ci_val + Kco) - Rd

                # RuBP-regeneration-limited rate
                Aj = 0.25 * J * (Ci_val - Gamma) / (Ci_val + 2 * Gamma) - Rd

                # Smooth minimum
                theta_smooth = 0.999
                A = (Ac + Aj - np.sqrt((Ac + Aj)**2 - 4 * theta_smooth * Ac * Aj)) / (2 * theta_smooth)

                A_result.append(A)

            return np.array(A_result).reshape(Ci_grid.shape)

        # Create figure with 2x3 subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Predicted vs Observed (1:1)
        ax1 = fig.add_subplot(gs[0, 0])

        # Plot black scatter points
        ax1.scatter(A_obs, A_pred, c='black', s=50, alpha=0.5,
                   edgecolors='black', linewidth=0.5, zorder=3)

        min_val = min(A_obs.min(), A_pred.min())
        max_val = max(A_obs.max(), A_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--',
                linewidth=2, label='1:1 line', zorder=1)

        ax1.set_xlabel('Measured A (μmol m⁻² s⁻¹)', fontsize=13)
        ax1.set_ylabel('Modeled A (μmol m⁻² s⁻¹)', fontsize=13)
        ax1.set_title('Predicted vs Observed', fontsize=13, fontweight='bold')
        ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax1.transAxes,
                fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax1.grid(True)
        ax1.legend(fontsize=9, loc='lower right')

        # Plot 2: A-Ci Response @ Q = 2000, T = 25°C
        ax2 = fig.add_subplot(gs[0, 1])

        # Generate smooth Ci curve
        Ci_range = np.linspace(0, 2000, 60)
        Q_grid = Q_mean * np.ones_like(Ci_range)
        T_grid = T_plot_K * np.ones_like(Ci_range)

        # Evaluate model
        A_model = evaluate_model_grid(Ci_range, Q_grid, T_grid)

        # Plot measured data first (lower zorder)
        ax2.scatter(Ci_obs, A_obs, c='black', s=30, alpha=0.5, label='Measured', zorder=1)

        # Plot model response curve on top (higher zorder)
        ax2.plot(Ci_range, A_model, 'r', linewidth=3, label='Model', zorder=3)

        ax2.set_xlabel('Ci (μmol mol⁻¹)', fontsize=13)
        ax2.set_ylabel('A (μmol m⁻² s⁻¹)', fontsize=13)
        ax2.set_title(f'A vs Ci @ Q = {Q_mean:.0f}, T = {T_plot_C:.1f}°C',
                     fontsize=13, fontweight='bold')
        ax2.set_ylim([0, max(A_obs.max(), A_model.max()) * 1.1])
        ax2.grid(True)
        ax2.legend(fontsize=10)

        # Plot 3: Light Response @ Ci = 2000 (saturating), T = 25°C
        ax3 = fig.add_subplot(gs[0, 2])

        # Generate smooth Q curve
        Q_range = np.linspace(0, 2000, 60)
        Ci_grid = 2000 * np.ones_like(Q_range)  # High Ci for light response
        T_grid = T_plot_K * np.ones_like(Q_range)

        # Evaluate model
        A_model = evaluate_model_grid(Ci_grid, Q_range, T_grid)

        # Plot measured data first (lower zorder)
        ax3.scatter(Q_obs, A_obs, c='black', s=30, alpha=0.5, label='Measured', zorder=1)

        # Plot model response curve on top (higher zorder)
        ax3.plot(Q_range, A_model, 'r', linewidth=3, label='Model', zorder=3)

        ax3.set_xlabel('Q (μmol m⁻² s⁻¹)', fontsize=13)
        ax3.set_ylabel('A (μmol m⁻² s⁻¹)', fontsize=13)
        ax3.set_title(f'A vs Q @ Ci = {Ci_grid[0]:.0f}, T = {T_plot_C:.1f}°C',
                     fontsize=13, fontweight='bold')
        ax3.set_ylim([0, max(A_obs.max(), A_model.max()) * 1.1])
        ax3.grid(True)
        ax3.legend(fontsize=10)

        # Plot 4: Temperature Response @ Q = 2000, Ci = 0.7*420
        ax4 = fig.add_subplot(gs[1, 0])

        # Generate smooth T curve
        T_range_C = np.linspace(10, 45, 60)
        T_range_K = T_range_C + 273.15
        Ci_grid = (0.7 * 420) * np.ones_like(T_range_K)  # 0.7 * atmospheric CO2
        Q_grid = Q_mean * np.ones_like(T_range_K)

        # Evaluate model
        A_model = evaluate_model_grid(Ci_grid, Q_grid, T_range_K)

        # Plot measured data first (lower zorder)
        ax4.scatter(Tleaf_obs - 273.15, A_obs, c='black', s=30, alpha=0.5, label='Measured', zorder=1)

        # Plot model response curve on top (higher zorder)
        ax4.plot(T_range_C, A_model, 'r', linewidth=3, label='Model', zorder=3)

        ax4.set_xlabel('T (°C)', fontsize=13)
        ax4.set_ylabel('A (μmol m⁻² s⁻¹)', fontsize=13)
        ax4.set_title(f'A vs T @ Q = {Q_mean:.0f}, Ci = {Ci_grid[0]:.0f}',
                     fontsize=13, fontweight='bold')
        ax4.set_ylim([0, max(A_obs.max(), A_model.max()) * 1.1])
        ax4.grid(True)
        ax4.legend(fontsize=10)

        # Plot 5: 3D Surface - A vs Ci vs Q @ T = 298.15 K
        ax5 = fig.add_subplot(gs[1, 1], projection='3d')

        # Create grid (use smaller grid for speed)
        Ci_3d = np.linspace(5, 2000, 30)
        Q_3d = np.linspace(0, 2000, 30)
        Ci_grid, Q_grid = np.meshgrid(Ci_3d, Q_3d)
        T_grid = 298.15 * np.ones_like(Ci_grid)

        # Evaluate model on grid
        A_surf = np.zeros_like(Ci_grid)
        for i in range(Ci_grid.shape[0]):
            for j in range(Ci_grid.shape[1]):
                A_surf[i,j] = evaluate_model_grid(
                    np.array([Ci_grid[i,j]]),
                    np.array([Q_grid[i,j]]),
                    np.array([T_grid[i,j]])
                )[0]

        # Plot surface
        ax5.plot_surface(Ci_grid, Q_grid, A_surf, cmap='YlGn',
                        edgecolor='none', alpha=0.8, zorder=1)

        # Filter measured data to T around 25°C (±2°C)
        T_filter_5 = np.abs((Tleaf_obs - 273.15) - 25.0) < 2.0
        ax5.scatter(Ci_obs[T_filter_5], Q_obs[T_filter_5], A_obs[T_filter_5],
                   c='r', s=30, alpha=0.7, label='Measured', zorder=3)

        ax5.set_xlabel('Ci (μmol mol⁻¹)', fontsize=13)
        ax5.set_ylabel('Q (μmol m⁻² s⁻¹)', fontsize=13)
        ax5.set_zlabel('A (μmol m⁻² s⁻¹)', fontsize=13)
        ax5.set_xticks([0, 1000, 2000])
        ax5.set_title('A vs Ci vs Q\n(T = 25°C)', fontsize=13, fontweight='bold')
        ax5.view_init(elev=5, azim=-10)
        ax5.legend(loc='upper right')

        # Plot 6: 3D Surface - A vs Ci vs T @ Q = 2000
        ax6 = fig.add_subplot(gs[1, 2], projection='3d')

        # Create grid (use smaller grid for speed)
        Ci_3d2 = np.linspace(100, 2000, 30)
        T_3d_K = np.linspace(10 + 273.15, 40 + 273.15, 30)
        Ci_grid2, T_grid_K = np.meshgrid(Ci_3d2, T_3d_K)
        Q_grid2 = Q_mean * np.ones_like(Ci_grid2)

        # Evaluate model on grid
        A_surf2 = np.zeros_like(Ci_grid2)
        for i in range(Ci_grid2.shape[0]):
            for j in range(Ci_grid2.shape[1]):
                A_surf2[i,j] = evaluate_model_grid(
                    np.array([Ci_grid2[i,j]]),
                    np.array([Q_grid2[i,j]]),
                    np.array([T_grid_K[i,j]])
                )[0]

        T_grid_C = T_grid_K - 273.15

        # Plot surface
        ax6.plot_surface(Ci_grid2, T_grid_C, A_surf2, cmap='YlGn',
                        edgecolor='none', alpha=0.5, zorder=1)

        # Filter measured data to Q around 2000 (>1900)
        Q_filter_6 = Q_obs > 1900
        ax6.scatter(Ci_obs[Q_filter_6], Tleaf_obs[Q_filter_6] - 273.15, A_obs[Q_filter_6],
                   c='r', s=30, alpha=0.7, label='Measured', zorder=3)

        ax6.set_xlabel('Ci (μmol mol⁻¹)', fontsize=13)
        ax6.set_ylabel('T (°C)', fontsize=13)
        ax6.set_zlabel('A (μmol m⁻² s⁻¹)', fontsize=13)
        ax6.set_xticks([0, 1000, 2000])
        ax6.set_title(f'A vs Ci vs T\n(Q = {Q_mean:.0f})', fontsize=13, fontweight='bold')
        ax6.view_init(elev=5, azim=-10)

        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save}")

        if show:
            plt.show()

        return fig
