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
                'units': 'μmol m⁻² s⁻¹',
                'description': 'Maximum Rubisco carboxylation rate at 25°C',
                'symbol': 'V_{cmax25}'
            },
            'Jmax25': {
                'default': 200.0,
                'bounds': (40.0, 600.0),
                'units': 'μmol m⁻² s⁻¹',
                'description': 'Maximum electron transport rate at 25°C',
                'symbol': 'J_{max25}'
            },
            'TPU25': {
                'default': 25.0,
                'bounds': (5.0, 100.0),
                'units': 'μmol m⁻² s⁻¹',
                'description': 'Triose phosphate utilization rate at 25°C',
                'symbol': 'TPU_{25}'
            },
            'Rd25': {
                'default': 1.5,
                'bounds': (0.0, 10.0),
                'units': 'μmol m⁻² s⁻¹',
                'description': 'Dark respiration rate at 25°C',
                'symbol': 'R_{d25}'
            },

            # Light response parameters
            'LightResponse.alpha': {
                'default': 0.9,
                'bounds': (0.0, 1.0),
                'units': 'mol e⁻ / mol photon',
                'description': 'Quantum yield of electron transport',
                'symbol': 'α'
            },
            'LightResponse.theta': {
                'default': 0.7,
                'bounds': (0.0, 1.0),
                'units': 'dimensionless',
                'description': 'Curvature factor for light response',
                'symbol': 'θ'
            },

            # Temperature response parameters - Activation energies
            'TempResponse.dHa_Vcmax': {
                'default': 73.0,
                'bounds': (50.0, 120.0),
                'units': 'kJ mol⁻¹',
                'description': 'Activation energy for Vcmax',
                'symbol': 'ΔHa_{Vcmax}'
            },
            'TempResponse.dHa_Jmax': {
                'default': 33.0,
                'bounds': (20.0, 80.0),
                'units': 'kJ mol⁻¹',
                'description': 'Activation energy for Jmax',
                'symbol': 'ΔHa_{Jmax}'
            },
            'TempResponse.dHa_TPU': {
                'default': 73.0,
                'bounds': (50.0, 120.0),
                'units': 'kJ mol⁻¹',
                'description': 'Activation energy for TPU',
                'symbol': 'ΔHa_{TPU}'
            },

            # Temperature response parameters - Optimal temperatures (peaked Arrhenius)
            'TempResponse.Topt_Vcmax': {
                'default': 311.15,
                'bounds': (298.15, 323.15),
                'units': 'K',
                'description': 'Optimal temperature for Vcmax',
                'symbol': 'T_{opt,Vcmax}'
            },
            'TempResponse.Topt_Jmax': {
                'default': 311.15,
                'bounds': (298.15, 323.15),
                'units': 'K',
                'description': 'Optimal temperature for Jmax',
                'symbol': 'T_{opt,Jmax}'
            },
            'TempResponse.Topt_TPU': {
                'default': 311.15,
                'bounds': (298.15, 323.15),
                'units': 'K',
                'description': 'Optimal temperature for TPU',
                'symbol': 'T_{opt,TPU}'
            },

            # Optional biochemical parameters
            'gm': {
                'default': 0.4,
                'bounds': (0.01, 2.0),
                'units': 'mol m⁻² s⁻¹ bar⁻¹',
                'description': 'Mesophyll conductance to CO₂',
                'symbol': 'g_m'
            },
            'Gamma25': {
                'default': 42.75,
                'bounds': (30.0, 60.0),
                'units': 'μmol mol⁻¹',
                'description': 'CO₂ compensation point at 25°C',
                'symbol': 'Γ*_{25}'
            },
            'Kc25': {
                'default': 404.9,
                'bounds': (200.0, 800.0),
                'units': 'μmol mol⁻¹',
                'description': 'Michaelis constant for CO₂ at 25°C',
                'symbol': 'K_{c25}'
            },
            'Ko25': {
                'default': 278.4,
                'bounds': (100.0, 500.0),
                'units': 'mmol mol⁻¹',
                'description': 'Michaelis constant for O₂ at 25°C',
                'symbol': 'K_{o25}'
            },
            'alphaG_r': {
                'default': 0.5,
                'bounds': (0.0, 1.0),
                'units': 'dimensionless',
                'description': 'Stoichiometric ratio of orthophosphate (Pi) consumption in oxygenation',
                'symbol': 'α_g'
            },
            'Rdratio': {
                'default': 0.015,
                'bounds': (0.0, 0.05),
                'units': 'dimensionless',
                'description': 'Rd as fraction of Vcmax',
                'symbol': 'R_d/V_{cmax}'
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

        # Helper function to evaluate model on grid using pure FvCB (no gm correction)
        # This matches temp_photorch/photorch/src/fvcb/evaluate.py approach
        def evaluate_model_grid(Ci_grid, Q_grid, T_grid):
            """
            Evaluate model on a grid using FvCB equations with Ci (not Cc).
            Uses smooth hyperbolic minimum as in legacy evaluateFvCB.
            No gm correction, no TPU limitation - only Ac and Aj.
            """
            n_points = len(Ci_grid.flatten())

            # Create temporary tensors
            Ci_tensor = torch.tensor(Ci_grid.flatten(), dtype=torch.float32)
            Q_tensor = torch.tensor(Q_grid.flatten(), dtype=torch.float32)
            T_tensor = torch.tensor(T_grid.flatten(), dtype=torch.float32)

            # Smooth hyperbolic minimum function
            def hmin(f1, f2):
                """Smooth minimum function to avoid sharp transitions"""
                theta_smooth = 0.999
                return (f1 + f2 - np.sqrt((f1 + f2)**2 - 4 * theta_smooth * f1 * f2)) / (2 * theta_smooth)

            # Get mean parameter values for evaluation
            with torch.no_grad():
                # Extract parameters
                params_dict = {}
                for name, param in self.core_model.named_parameters():
                    if param.numel() > 1:
                        params_dict[name] = param.mean().item()
                    else:
                        params_dict[name] = param.item()

                # Get biochemical parameters
                from .fvcb_core_legacy import allparameters
                ap = allparameters()

                # Calculate photosynthesis for each point
                A_result = []
                for i in range(n_points):
                    Ci_val = Ci_tensor[i].item()
                    Q_val = Q_tensor[i].item()
                    T_val = T_tensor[i].item()

                    # Get Vcmax, Jmax, Rd at reference temperature
                    Vcmax25 = params_dict.get('Vcmax25', 100.0)
                    Jmax25 = params_dict.get('Jmax25', 200.0)
                    Rd25 = params_dict.get('Rd25', 1.5)

                    # Temperature corrections
                    R = 0.0083144598  # kJ/(mol·K)
                    Tref = 298.15

                    # Get temperature response parameters
                    dHa_Vcmax = params_dict.get('TempResponse.dHa_Vcmax', 73.0)
                    dHa_Jmax = params_dict.get('TempResponse.dHa_Jmax', 33.0)

                    # Check if using peaked Arrhenius (temp_response=2)
                    has_Topt = 'TempResponse.Topt_Vcmax' in params_dict

                    if has_Topt:
                        # Peaked Arrhenius temperature response
                        Topt_Vcmax = params_dict.get('TempResponse.Topt_Vcmax', 311.15)
                        Topt_Jmax = params_dict.get('TempResponse.Topt_Jmax', 311.15)
                        dHd_Vcmax = ap.dHd_Vcmax.item()
                        dHd_Jmax = ap.dHd_Jmax.item()

                        # Simple Arrhenius first
                        Vcmax_arr = Vcmax25 * np.exp(dHa_Vcmax * (T_val - Tref) / (R * T_val * Tref))
                        Jmax_arr = Jmax25 * np.exp(dHa_Jmax * (T_val - Tref) / (R * T_val * Tref))

                        # Apply deactivation (peaked Arrhenius)
                        def peaked_arrhenius(k_arr, dHa, dHd, Topt):
                            dHd_R = dHd / R
                            dHd_dHa = dHd / dHa
                            dHd_dHa = max(dHd_dHa, 1.0001)
                            log_dHd_dHa = np.log(dHd_dHa - 1)
                            rec_Topt = 1 / Topt
                            rec_Tref = 1 / Tref
                            rec_Tval = 1 / T_val

                            numerator = 1 + np.exp(dHd_R * (rec_Topt - rec_Tref) - log_dHd_dHa)
                            denominator = 1 + np.exp(dHd_R * (rec_Topt - rec_Tval) - log_dHd_dHa)
                            return k_arr * numerator / denominator

                        Vcmax = peaked_arrhenius(Vcmax_arr, dHa_Vcmax, dHd_Vcmax, Topt_Vcmax)
                        Jmax = peaked_arrhenius(Jmax_arr, dHa_Jmax, dHd_Jmax, Topt_Jmax)
                    else:
                        # Simple Arrhenius temperature response
                        Vcmax = Vcmax25 * np.exp(dHa_Vcmax * (T_val - Tref) / (R * T_val * Tref))
                        Jmax = Jmax25 * np.exp(dHa_Jmax * (T_val - Tref) / (R * T_val * Tref))

                    # Rd uses simple Arrhenius
                    Rd = Rd25 * np.exp(46.39 * (T_val - Tref) / (R * T_val * Tref))

                    # Get Kc, Ko, Gamma
                    Kc = ap.Kc25 * np.exp(ap.dHa_Kc * (T_val - Tref) / (R * T_val * Tref))
                    Ko = ap.Ko25 * np.exp(ap.dHa_Ko * (T_val - Tref) / (R * T_val * Tref))
                    Gamma = ap.Gamma25 * np.exp(ap.dHa_Gamma * (T_val - Tref) / (R * T_val * Tref))

                    # Calculate J from light response
                    alpha = params_dict.get('LightResponse.alpha', 0.9)
                    theta = params_dict.get('LightResponse.theta', 0.7) if hasattr(self.core_model.LightResponse, 'theta') else 0.7
                    J = (alpha * Q_val + Jmax - np.sqrt((alpha * Q_val + Jmax)**2 - 4 * theta * alpha * Q_val * Jmax)) / (2 * theta)

                    # Calculate RuBisCO-limited rate (vr in legacy)
                    Kco = Kc * (1 + ap.oxy / Ko)
                    Ac = Vcmax * ((Ci_val - Gamma) / (Ci_val + Kco)) - Rd

                    # Calculate RuBP-limited rate (jr in legacy)
                    # Note: 0.25 * J = J/4 (electron transport to CO2 fixation ratio)
                    Aj = 0.25 * J * ((Ci_val - Gamma) / (Ci_val + 2 * Gamma)) - Rd

                    # Use smooth hyperbolic minimum (no hard min, no TPU)
                    A = hmin(Ac, Aj)

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
