"""Main fitting function for PhyTorch."""

from typing import Dict, Optional, Union
import pandas as pd

from phytorch.models.base import Model
from phytorch.core.result import FitResult
from phytorch.core.optimizers.scipy_optimizer import fit_with_scipy


def fit(
    model: Model,
    data: Union[Dict, pd.DataFrame],
    options: Optional[Dict] = None
) -> FitResult:
    """Fit a physiological model to data.

    This is the unified interface for all PhyTorch model fitting.

    Args:
        model: Model instance from phytorch.models.*
              (e.g., phytorch.models.hydraulics.Sigmoid())
        data: Input data as dict or DataFrame
              Keys/columns must match model's required_data()
              Example: {'psi': np.array([...]), 'K': np.array([...])}
        options: Fitting configuration (optional):
            - method: 'scipy', 'adam', 'lbfgs', or 'auto' (default: 'auto')
            - fit_parameters: List of parameters to fit (default: all)
            - fixed_parameters: Dict of parameters to keep fixed
            - bounds: Dict of (lower, upper) bounds per parameter
            - initial_guess: Dict of initial parameter values
            - max_iterations: Maximum optimization iterations
            - learning_rate: Learning rate (for PyTorch optimizers)
            - min_loss: Early stopping threshold (for PyTorch)
            - device: 'cpu' or 'cuda' (for PyTorch)
            - verbose: Print progress (default: True)
            - ftol: Function tolerance (for scipy)
            - xtol: Parameter tolerance (for scipy)

    Returns:
        FitResult with fitted parameters, predictions, and diagnostics

    Example:
        >>> from phytorch import fit
        >>> from phytorch.models.hydraulics import Sigmoid
        >>> import numpy as np
        >>>
        >>> # Prepare data
        >>> data = {
        ...     'psi': np.array([-0.5, -1.0, -1.5, -2.0, -2.5]),
        ...     'K': np.array([10.2, 8.5, 5.2, 2.1, 0.8])
        ... }
        >>>
        >>> # Fit model
        >>> result = fit(Sigmoid(), data)
        >>> print(result.summary())
        >>>
        >>> # Make predictions
        >>> psi_new = np.linspace(-3, 0, 100)
        >>> K_pred = result.predict({'psi': psi_new})

    Example with options:
        >>> result = fit(
        ...     model=Sigmoid(),
        ...     data=data,
        ...     options={
        ...         'method': 'scipy',
        ...         'bounds': {
        ...             'Kmax': (8.0, 12.0),  # Constrain Kmax
        ...             'psi50': (-2.5, -0.5)
        ...         },
        ...         'fixed_parameters': {'s': 3.0},  # Fix steepness
        ...         'verbose': True
        ...     }
        ... )
    """
    options = options or {}

    # Convert DataFrame to dict if needed
    if isinstance(data, pd.DataFrame):
        data = {col: data[col].values for col in data.columns}

    # Determine optimization method
    method = options.get('method', 'auto')

    if method == 'auto':
        # Auto-select based on model type
        # Check if model has use_torch_optimizer flag (for complex coupled models)
        if hasattr(model, 'use_torch_optimizer') and model.use_torch_optimizer:
            method = 'torch'
        else:
            # Default to scipy (fast and robust for empirical models)
            method = 'scipy'

    # Dispatch to appropriate optimizer
    if method == 'scipy':
        return fit_with_scipy(model, data, options)
    elif method in ['torch', 'adam']:
        from phytorch.core.optimizers.torch_optimizer import fit_with_torch
        return fit_with_torch(model, data, options)
    else:
        raise ValueError(
            f"Unknown optimization method: '{method}'. "
            f"Choose from: 'scipy', 'torch', 'adam', 'auto'"
        )
