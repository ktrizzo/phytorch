"""Result class for model fitting."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import os


@dataclass
class FitResult:
    """Results from fitting a physiological model to data.

    Attributes:
        model: The fitted model instance
        parameters: Dict of fitted parameter values
        data: Original input data
        predictions: Model predictions at data points
        residuals: Prediction errors (observed - predicted)
        loss: Final loss value (RSS for scipy, custom for PyTorch)
        r_squared: Coefficient of determination (if applicable)
        converged: Whether optimization converged successfully
        iterations: Number of iterations used
        optimizer_info: Additional optimizer-specific information
        covariance: Parameter covariance matrix (if available, from scipy)
        fit_options: Options used for fitting
        fit_time: Timestamp when fit was performed
    """
    model: object  # phytorch.models.base.Model
    parameters: Dict[str, float]
    data: Dict
    predictions: np.ndarray
    residuals: np.ndarray
    loss: float
    r_squared: Optional[float]
    converged: bool
    iterations: int
    optimizer_info: Dict
    covariance: Optional[np.ndarray]
    fit_options: Dict = field(default_factory=dict)
    fit_time: Optional[datetime] = None

    def predict(self, data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """Make predictions with fitted model on new data.

        Args:
            data: New input data as dict or DataFrame
                  Must contain all input fields required by model
                  (output field is not required for prediction)

        Returns:
            Model predictions as numpy array

        Example:
            >>> psi_new = np.linspace(-3, 0, 100)
            >>> K_pred = result.predict({'psi': psi_new})
        """
        # Convert DataFrame to dict if needed
        if isinstance(data, pd.DataFrame):
            data = {col: data[col].values for col in data.columns}

        # Make predictions
        # Note: We don't validate here because required_data() includes
        # the output variable, which we don't have when predicting
        return self.model.forward(data, self.parameters)

    def summary(self) -> str:
        """Generate formatted summary of fit results.

        Returns:
            Multi-line string with fitted parameters and diagnostics
        """
        lines = [f"Fit Results: {self.model.__class__.__name__}"]
        lines.append("=" * 60)
        lines.append("\nFitted Parameters:")

        param_info = self.model.parameter_info()
        for name, value in self.parameters.items():
            info = param_info[name]
            units = info.get('units', '')
            symbol = info.get('symbol', name)
            lines.append(f"  {symbol:20s} = {value:12.6f}  {units}")

        lines.append("\nGoodness of Fit:")
        lines.append(f"  Loss             = {self.loss:.6f}")
        if self.r_squared is not None:
            lines.append(f"  R²               = {self.r_squared:.6f}")
        lines.append(f"  Converged        = {self.converged}")
        lines.append(f"  Iterations       = {self.iterations}")

        # Add optimizer-specific info
        if self.optimizer_info:
            lines.append("\nOptimizer Info:")
            for key, value in self.optimizer_info.items():
                lines.append(f"  {key:20s} = {value}")

        return "\n".join(lines)

    def __repr__(self):
        """Short representation of fit results."""
        params_str = ", ".join(f"{k}={v:.3f}" for k, v in self.parameters.items())
        loss_str = f"loss={self.loss:.4f}" if self.loss < 1e3 else f"loss={self.loss:.2e}"
        return f"FitResult({self.model.__class__.__name__}, {params_str}, {loss_str})"

    def parameter_uncertainties(self) -> Optional[Dict[str, float]]:
        """Calculate parameter standard errors from covariance matrix.

        Returns:
            Dict of parameter standard errors, or None if covariance unavailable

        Note:
            Only available when using scipy optimizer with successful fit.
        """
        if self.covariance is None:
            return None

        # Standard error = sqrt(diagonal of covariance matrix)
        param_names = list(self.parameters.keys())
        std_errors = np.sqrt(np.diag(self.covariance))

        return dict(zip(param_names, std_errors))

    def correlation_matrix(self) -> Optional[np.ndarray]:
        """Calculate parameter correlation matrix from covariance.

        Returns:
            Correlation matrix, or None if covariance unavailable

        Note:
            Correlation matrix element (i,j) = cov(i,j) / (std(i) * std(j))
        """
        if self.covariance is None:
            return None

        # Convert covariance to correlation
        std = np.sqrt(np.diag(self.covariance))
        corr = self.covariance / np.outer(std, std)

        return corr

    def plot(self, save: Optional[str] = None, show: bool = True):
        """Plot model fit results.

        Creates diagnostic plots:
        - For 1D models (x→y): Predicted vs Observed, and Model Fit curve
        - For photosynthesis models: Special 3D layout with surface plots
        - For other multi-D models: Predicted vs Observed, plus response curves
        - For models with custom plot: Delegates to model.plot()

        Args:
            save: Optional filename to save figure (e.g., 'fit.png')
            show: Whether to display the plot (default True)

        Example:
            >>> result.plot()  # Display plots
            >>> result.plot(save='figure.png', show=False)  # Save without display
        """
        # Check if model has custom plot method
        if hasattr(self.model, 'plot') and callable(self.model.plot):
            return self.model.plot(self.data, self.parameters, show=show, save=save)

        # Get required data fields (inputs + output)
        required = self.model.required_data()

        # Detect input vs output variables
        # For most models, last field is output, rest are inputs
        if len(required) == 2:
            # Simple 1D case
            input_var = required[0]
            output_var = required[1]
            self._plot_1d(input_var, output_var, save, show)
        else:
            # Multi-dimensional case
            input_vars = required[:-1]
            output_var = required[-1]

            # Check if this is a photosynthesis model (has Ci, Q, T or Tleaf)
            is_photosynthesis = self._is_photosynthesis_model(input_vars)

            if is_photosynthesis:
                self._plot_photosynthesis(input_vars, output_var, save, show)
            else:
                self._plot_multid(input_vars, output_var, save, show)

    def _plot_1d(self, input_var: str, output_var: str, save: Optional[str], show: bool):
        """Plot 1D model (single input variable)."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Get data
        x_data = self.data[input_var]
        y_data = self.data[output_var]
        y_pred = self.predictions

        # Plot 1: Predicted vs Observed
        ax = axes[0]
        ax.scatter(y_data, y_pred, c='black', alpha=0.6, edgecolors='k', linewidth=0.5)

        # 1:1 line
        y_min = min(y_data.min(), y_pred.min())
        y_max = max(y_data.max(), y_pred.max())
        ax.plot([y_min, y_max], [y_min, y_max], 'k--', lw=1, label='1:1 line')

        ax.set_xlabel(f'Observed {output_var}')
        ax.set_ylabel(f'Predicted {output_var}')
        ax.set_title('Predicted vs Observed')
        if self.r_squared is not None:
            ax.text(0.05, 0.95, f'R² = {self.r_squared:.4f}',
                   transform=ax.transAxes, va='top')
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 2: Model fit
        ax = axes[1]

        # Sort data by input for plotting
        sort_idx = np.argsort(x_data)
        ax.scatter(x_data, y_data, c='black', alpha=0.6, label='Observed',
                  edgecolors='k', linewidth=0.5)

        # Create smooth prediction curve
        x_min, x_max = x_data.min(), x_data.max()
        x_range = x_max - x_min
        x_smooth = np.linspace(x_min - 0.05*x_range, x_max + 0.05*x_range, 200)
        y_smooth = self.predict({input_var: x_smooth})

        ax.plot(x_smooth, y_smooth, 'r-', lw=2, label='Model fit')

        ax.set_xlabel(input_var)
        ax.set_ylabel(output_var)
        ax.set_title(f'{self.model.__class__.__name__} Model Fit')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def _plot_multid(self, input_vars: list, output_var: str, save: Optional[str], show: bool):
        """Plot multi-dimensional model (multiple input variables)."""
        n_inputs = len(input_vars)
        fig, axes = plt.subplots(1, n_inputs + 1, figsize=(5 * (n_inputs + 1), 5))

        # Get data
        y_data = self.data[output_var]
        y_pred = self.predictions

        # Plot 1: Predicted vs Observed
        ax = axes[0]
        ax.scatter(y_data, y_pred, c='black', alpha=0.6, edgecolors='k', linewidth=0.5)

        # 1:1 line
        y_min = min(y_data.min(), y_pred.min())
        y_max = max(y_data.max(), y_pred.max())
        ax.plot([y_min, y_max], [y_min, y_max], 'k--', lw=1, label='1:1 line')

        ax.set_xlabel(f'Observed {output_var}')
        ax.set_ylabel(f'Predicted {output_var}')
        ax.set_title('Predicted vs Observed')
        if self.r_squared is not None:
            ax.text(0.05, 0.95, f'R² = {self.r_squared:.4f}',
                   transform=ax.transAxes, va='top')
        ax.legend()
        ax.grid(alpha=0.3)

        # Calculate median values for holding other variables constant
        median_values = {var: np.median(self.data[var]) for var in input_vars}

        # Plot 2+: One plot per input variable
        for i, var in enumerate(input_vars):
            ax = axes[i + 1]

            # Get this variable's data
            x_data = self.data[var]

            # Plot observed data
            ax.scatter(x_data, y_data, c='black', alpha=0.6, label='Observed',
                      edgecolors='k', linewidth=0.5)

            # Create prediction curve: vary this variable, hold others at median
            x_min, x_max = x_data.min(), x_data.max()
            x_range = x_max - x_min
            x_smooth = np.linspace(x_min - 0.05*x_range, x_max + 0.05*x_range, 200)

            # Build prediction data: this var varies, others constant
            pred_data = {v: np.full_like(x_smooth, median_values[v]) for v in input_vars}
            pred_data[var] = x_smooth

            y_smooth = self.predict(pred_data)

            ax.plot(x_smooth, y_smooth, 'r-', lw=2, label='Model (others at median)')

            ax.set_xlabel(var)
            ax.set_ylabel(output_var)
            ax.set_title(f'{output_var} vs {var}')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.suptitle(f'{self.model.__class__.__name__} Model Fit', fontsize=14, y=1.02)
        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def _is_photosynthesis_model(self, input_vars: list) -> bool:
        """Check if this is a photosynthesis model based on variable names."""
        input_set = set(input_vars)
        # Photosynthesis models have Ci, Q, and T or Tleaf
        has_ci = 'Ci' in input_set
        has_q = 'Q' in input_set
        has_t = 'T' in input_set or 'Tleaf' in input_set
        return has_ci and has_q and has_t

    def _plot_photosynthesis(self, input_vars: list, output_var: str, save: Optional[str], show: bool):
        """Plot photosynthesis model with 2D response curves and 3D surfaces."""
        # Create figure with custom layout: 2 rows, 4 columns
        # Row 1: 1:1, A vs Ci, A vs Q, A vs T
        # Row 2: (empty), 3D Ci-Q-A (spans 2 cols), 3D Ci-T-A (spans 2 cols)
        fig = plt.figure(figsize=(20, 10))

        # Get data
        y_data = self.data[output_var]
        y_pred = self.predictions

        # Identify which variable is which
        ci_var = 'Ci' if 'Ci' in input_vars else None
        q_var = 'Q' if 'Q' in input_vars else None
        t_var = 'Tleaf' if 'Tleaf' in input_vars else ('T' if 'T' in input_vars else None)

        if not (ci_var and q_var and t_var):
            # Fallback to generic multid plot
            self._plot_multid(input_vars, output_var, save, show)
            return

        # Calculate median values for holding variables constant
        median_values = {var: np.median(self.data[var]) for var in input_vars}

        # Row 1, Col 1: Predicted vs Observed
        ax1 = plt.subplot(2, 4, 1)
        ax1.scatter(y_data, y_pred, c='black', alpha=0.6, edgecolors='k', linewidth=0.5)
        y_min = min(y_data.min(), y_pred.min())
        y_max = max(y_data.max(), y_pred.max())
        ax1.plot([y_min, y_max], [y_min, y_max], 'k--', lw=1, label='1:1 line')
        ax1.set_xlabel(f'Observed {output_var}')
        ax1.set_ylabel(f'Predicted {output_var}')
        ax1.set_title('Predicted vs Observed')
        if self.r_squared is not None:
            ax1.text(0.05, 0.95, f'R² = {self.r_squared:.4f}',
                    transform=ax1.transAxes, va='top')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Row 1, Col 2: A vs Ci (Q, T constant)
        ax2 = plt.subplot(2, 4, 2)
        ci_data = self.data[ci_var]
        ax2.scatter(ci_data, y_data, c='black', alpha=0.6, label='Observed', edgecolors='k', linewidth=0.5)

        ci_smooth = np.linspace(ci_data.min() * 0.95, ci_data.max() * 1.05, 200)
        pred_data = {v: np.full_like(ci_smooth, median_values[v]) for v in input_vars}
        pred_data[ci_var] = ci_smooth
        a_smooth = self.predict(pred_data)
        ax2.plot(ci_smooth, a_smooth, 'r-', lw=2, label=f'Model (Q={median_values[q_var]:.0f}, T={median_values[t_var]:.1f})')
        ax2.set_xlabel(ci_var)
        ax2.set_ylabel(output_var)
        ax2.set_title(f'{output_var} vs {ci_var}')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # Row 1, Col 3: A vs Q (Ci, T constant)
        ax3 = plt.subplot(2, 4, 3)
        q_data = self.data[q_var]
        ax3.scatter(q_data, y_data, c='black', alpha=0.6, label='Observed', edgecolors='k', linewidth=0.5)

        q_smooth = np.linspace(q_data.min() * 0.95, q_data.max() * 1.05, 200)
        pred_data = {v: np.full_like(q_smooth, median_values[v]) for v in input_vars}
        pred_data[q_var] = q_smooth
        a_smooth = self.predict(pred_data)
        ax3.plot(q_smooth, a_smooth, 'r-', lw=2, label=f'Model (Ci={median_values[ci_var]:.0f}, T={median_values[t_var]:.1f})')
        ax3.set_xlabel(q_var)
        ax3.set_ylabel(output_var)
        ax3.set_title(f'{output_var} vs {q_var}')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

        # Row 1, Col 4: A vs T (Ci, Q constant)
        ax4 = plt.subplot(2, 4, 4)
        t_data = self.data[t_var]
        ax4.scatter(t_data, y_data, c='black', alpha=0.6, label='Observed', edgecolors='k', linewidth=0.5)

        t_smooth = np.linspace(t_data.min() - 1, t_data.max() + 1, 200)
        pred_data = {v: np.full_like(t_smooth, median_values[v]) for v in input_vars}
        pred_data[t_var] = t_smooth
        a_smooth = self.predict(pred_data)
        ax4.plot(t_smooth, a_smooth, 'r-', lw=2, label=f'Model (Ci={median_values[ci_var]:.0f}, Q={median_values[q_var]:.0f})')
        ax4.set_xlabel(t_var)
        ax4.set_ylabel(output_var)
        ax4.set_title(f'{output_var} vs {t_var}')
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)

        # Row 2, Cols 2-3: 3D surface Ci vs Q vs A (T constant)
        ax5 = fig.add_subplot(2, 4, (6, 7), projection='3d')

        # Create meshgrid for surface
        ci_range = np.linspace(ci_data.min(), ci_data.max(), 30)
        q_range = np.linspace(q_data.min(), q_data.max(), 30)
        Ci_grid, Q_grid = np.meshgrid(ci_range, q_range)

        # Predict A for each point on grid (T constant at median)
        A_grid = np.zeros_like(Ci_grid)
        for i in range(Ci_grid.shape[0]):
            for j in range(Ci_grid.shape[1]):
                pred_data = {
                    ci_var: np.array([Ci_grid[i, j]]),
                    q_var: np.array([Q_grid[i, j]]),
                    t_var: np.array([median_values[t_var]])
                }
                A_grid[i, j] = self.predict(pred_data)[0]

        # Plot surface
        ax5.plot_surface(Ci_grid, Q_grid, A_grid, alpha=0.6, cmap='viridis', edgecolor='none')

        # Overlay observed data points
        ax5.scatter(self.data[ci_var], self.data[q_var], y_data,
                   c='black', s=30, alpha=0.8, edgecolors='k', linewidth=0.5)

        ax5.set_xlabel(ci_var)
        ax5.set_ylabel(q_var)
        ax5.set_zlabel(output_var)
        ax5.set_title(f'{ci_var} vs {q_var} vs {output_var} (T={median_values[t_var]:.1f})')
        ax5.view_init(elev=30, azim=15)

        # Row 2, Cols 4: 3D surface Ci vs T vs A (Q constant)
        ax6 = fig.add_subplot(2, 4, 8, projection='3d')

        # Create meshgrid for surface
        t_range = np.linspace(t_data.min(), t_data.max(), 30)
        Ci_grid2, T_grid = np.meshgrid(ci_range, t_range)

        # Predict A for each point on grid (Q constant at median)
        A_grid2 = np.zeros_like(Ci_grid2)
        for i in range(Ci_grid2.shape[0]):
            for j in range(Ci_grid2.shape[1]):
                pred_data = {
                    ci_var: np.array([Ci_grid2[i, j]]),
                    q_var: np.array([median_values[q_var]]),
                    t_var: np.array([T_grid[i, j]])
                }
                A_grid2[i, j] = self.predict(pred_data)[0]

        # Plot surface
        ax6.plot_surface(Ci_grid2, T_grid, A_grid2, alpha=0.6, cmap='viridis', edgecolor='none')

        # Overlay observed data points
        ax6.scatter(self.data[ci_var], self.data[t_var], y_data,
                   c='black', s=30, alpha=0.8, edgecolors='k', linewidth=0.5)

        ax6.set_xlabel(ci_var)
        ax6.set_ylabel(t_var)
        ax6.set_zlabel(output_var)
        ax6.set_title(f'{ci_var} vs {t_var} vs {output_var} (Q={median_values[q_var]:.0f})')
        ax6.view_init(elev=30, azim=15)

        plt.suptitle(f'{self.model.__class__.__name__} Photosynthesis Model', fontsize=16, y=0.98)
        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def write(self, filepath: Optional[str] = None):
        """Write fit results and data to CSV files.

        Creates two files:
        1. Data file: Contains the original input data used for fitting
        2. Results file: Contains fitted parameters, error metrics, and metadata

        The results file references the data file, creating a complete record
        of the fit that can be reloaded or shared.

        Args:
            filepath: Path for the results CSV file (e.g., 'results.csv')
                     If None, auto-generates: '{ModelName}_{datetime}_results.csv'
                     The data file will be created as '{basename}_data.csv'

        Example:
            >>> result = fit(model, data)
            >>> result.write()  # Auto-generates filename
            # Creates: MED2011_20250205_143022_results.csv and MED2011_20250205_143022_results_data.csv

            >>> result.write('my_fit_results.csv')  # Custom filename
            # Creates: my_fit_results.csv and my_fit_results_data.csv
        """
        # Generate default filepath if not provided
        if filepath is None:
            model_name = self.model.__class__.__name__
            if self.fit_time:
                datetime_str = self.fit_time.strftime('%Y%m%d_%H%M%S')
            else:
                datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"{model_name}_{datetime_str}_results.csv"

        # Parse filepath
        directory = os.path.dirname(filepath) or '.'
        basename = os.path.basename(filepath)
        name, ext = os.path.splitext(basename)
        if not ext:
            ext = '.csv'
            basename = name + ext

        # Create data filename
        data_filename = f"{name}_data{ext}"
        data_filepath = os.path.join(directory, data_filename)

        # Write data file
        data_df = pd.DataFrame(self.data)
        data_df['predictions'] = self.predictions
        data_df['residuals'] = self.residuals
        data_df.to_csv(data_filepath, index=False)

        # Prepare results data
        results_data = []

        # Add metadata
        results_data.append({'Category': 'Metadata', 'Parameter': 'Model', 'Value': self.model.__class__.__name__, 'Units': '', 'Notes': ''})
        results_data.append({'Category': 'Metadata', 'Parameter': 'Data File', 'Value': data_filename, 'Units': '', 'Notes': ''})

        if self.fit_time:
            time_str = self.fit_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        results_data.append({'Category': 'Metadata', 'Parameter': 'Fit Date/Time', 'Value': time_str, 'Units': '', 'Notes': ''})

        # Add fitted parameters
        param_info = self.model.parameter_info()
        fitted_params = self.optimizer_info.get('fitted_parameters', []) if self.optimizer_info else []

        # Normalize parameter names for matching (remove module prefixes like 'core_model.')
        def normalize_name(name):
            # Map internal names to output names
            name_map = {
                'core_model.LightResponse.alpha': 'alpha',
                'core_model.LightResponse.theta': 'theta',
                'core_model.TempResponse.dHa_Vcmax': 'Vcmax_dHa',
                'core_model.TempResponse.dHa_Jmax': 'Jmax_dHa',
                'core_model.TempResponse.dHa_TPU': 'TPU_dHa',
                'core_model.TempResponse.dHa_Rd': 'Rd_dHa',
                'core_model.TempResponse.Topt_Vcmax': 'Vcmax_Topt',
                'core_model.TempResponse.Topt_Jmax': 'Jmax_Topt',
                'core_model.TempResponse.Topt_TPU': 'TPU_Topt',
            }
            return name_map.get(name, name.split('.')[-1])

        fitted_param_names_normalized = [normalize_name(n) for n in fitted_params]

        for param_name, param_value in self.parameters.items():
            info = param_info.get(param_name, {})
            units = info.get('units', '')
            description = info.get('description', '')
            symbol = info.get('symbol', param_name)
            is_fitted = param_name in fitted_param_names_normalized
            category = 'Fitted Parameters' if is_fitted else 'Fixed Parameters'
            notes = f'{symbol}: {description}' if description else symbol
            if not is_fitted:
                notes = f'[FIXED] {notes}'
            results_data.append({
                'Category': category,
                'Parameter': param_name,
                'Value': param_value,
                'Units': units,
                'Notes': notes
            })

        # Add parameter uncertainties if available
        uncertainties = self.parameter_uncertainties()
        if uncertainties:
            for param_name, std_error in uncertainties.items():
                results_data.append({
                    'Category': 'Parameter Uncertainties',
                    'Parameter': f'{param_name}_SE',
                    'Value': std_error,
                    'Units': param_info.get(param_name, {}).get('units', ''),
                    'Notes': f'Standard error for {param_name}'
                })

        # Add error metrics
        results_data.append({'Category': 'Error Metrics', 'Parameter': 'Loss (RSS)', 'Value': self.loss, 'Units': '', 'Notes': 'Residual sum of squares'})
        if self.r_squared is not None:
            results_data.append({'Category': 'Error Metrics', 'Parameter': 'R²', 'Value': self.r_squared, 'Units': '', 'Notes': 'Coefficient of determination'})
        results_data.append({'Category': 'Error Metrics', 'Parameter': 'RMSE', 'Value': np.sqrt(self.loss / len(self.residuals)), 'Units': '', 'Notes': 'Root mean squared error'})

        # Add convergence info
        results_data.append({'Category': 'Convergence', 'Parameter': 'Converged', 'Value': str(self.converged), 'Units': '', 'Notes': ''})
        results_data.append({'Category': 'Convergence', 'Parameter': 'Iterations', 'Value': self.iterations, 'Units': '', 'Notes': ''})

        # Add fit options
        if self.fit_options:
            for option_name, option_value in self.fit_options.items():
                # Skip complex objects
                if isinstance(option_value, (str, int, float, bool, type(None))):
                    results_data.append({
                        'Category': 'Fit Options',
                        'Parameter': option_name,
                        'Value': str(option_value),
                        'Units': '',
                        'Notes': ''
                    })
                elif isinstance(option_value, dict):
                    # Flatten dict options
                    for sub_key, sub_value in option_value.items():
                        if isinstance(sub_value, (str, int, float, bool, type(None))):
                            results_data.append({
                                'Category': 'Fit Options',
                                'Parameter': f'{option_name}.{sub_key}',
                                'Value': str(sub_value),
                                'Units': '',
                                'Notes': ''
                            })

        # Add optimizer info
        if self.optimizer_info:
            for info_name, info_value in self.optimizer_info.items():
                if isinstance(info_value, (str, int, float, bool, type(None))):
                    results_data.append({
                        'Category': 'Optimizer Info',
                        'Parameter': info_name,
                        'Value': str(info_value),
                        'Units': '',
                        'Notes': ''
                    })

        # Write results file
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filepath, index=False)

        print(f"Results written to: {filepath}")
        print(f"Data written to: {data_filepath}")
