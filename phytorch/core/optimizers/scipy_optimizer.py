"""Scipy-based optimizer for PhyTorch models."""

from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime

from phytorch.core.result import FitResult


def fit_with_scipy(
    model,
    data: Dict,
    options: Optional[Dict] = None
) -> FitResult:
    """Fit model using scipy.optimize.curve_fit.

    Args:
        model: PhyTorch Model instance
        data: Input data dict
        options: Fitting options
            - fit_parameters: List of parameters to fit (default: all)
            - fixed_parameters: Dict of parameters to keep fixed
            - bounds: Dict of (lower, upper) bounds per parameter
            - initial_guess: Dict of initial parameter values
            - max_iterations: Maximum function evaluations (maxfev)
            - ftol: Function tolerance
            - xtol: Parameter tolerance
            - verbose: Print progress

    Returns:
        FitResult with fitted parameters and diagnostics
    """
    options = options or {}

    # Validate data
    model.validate_data(data)

    # Get parameter info
    param_info = model.parameter_info()
    all_params = list(param_info.keys())

    # Handle fixed vs fitted parameters
    fixed_params = options.get('fixed_parameters', {})
    fit_params_list = options.get('fit_parameters', None)

    if fit_params_list is None:
        # Fit all parameters except those explicitly fixed
        fit_params = [p for p in all_params if p not in fixed_params]
    else:
        # Fit only specified parameters
        fit_params = fit_params_list

    # Get observed data (use last required field as dependent variable)
    required_fields = model.required_data()
    y_field = required_fields[-1]  # Always use last field as dependent variable
    y_obs = np.asarray(data[y_field])

    # Get initial guess
    if 'initial_guess' in options:
        p0_dict = options['initial_guess']
    else:
        p0_dict = model.initial_guess(data)

    # Construct initial guess array for fitted parameters
    p0 = [p0_dict[p] for p in fit_params]

    # Construct bounds
    if 'bounds' in options:
        custom_bounds = options['bounds']
        lower = [custom_bounds.get(p, param_info[p]['bounds'])[0] for p in fit_params]
        upper = [custom_bounds.get(p, param_info[p]['bounds'])[1] for p in fit_params]
    else:
        lower = [param_info[p]['bounds'][0] for p in fit_params]
        upper = [param_info[p]['bounds'][1] for p in fit_params]

    bounds = (lower, upper)

    # Clip initial guess to bounds
    p0 = [np.clip(p0_val, lb, ub) for p0_val, lb, ub in zip(p0, lower, upper)]

    # Create wrapper function for curve_fit
    def model_func(x_dummy, *fit_param_values):
        # Combine fitted and fixed parameters
        params = dict(fixed_params)
        for name, value in zip(fit_params, fit_param_values):
            params[name] = value

        # Compute predictions
        return model.forward(data, params)

    # Dummy x for curve_fit (not used, but required by API)
    x_dummy = np.arange(len(y_obs))

    # Run optimization
    # Build kwargs for curve_fit
    curve_fit_kwargs = {
        'p0': p0,
        'bounds': bounds,
        'maxfev': options.get('max_iterations', 10000)
    }

    # Only add tolerances if explicitly specified (use scipy defaults otherwise)
    if 'ftol' in options:
        curve_fit_kwargs['ftol'] = options['ftol']
    if 'xtol' in options:
        curve_fit_kwargs['xtol'] = options['xtol']

    try:
        popt, pcov = curve_fit(model_func, x_dummy, y_obs, **curve_fit_kwargs)
        converged = True
    except RuntimeError as e:
        # Optimization failed to converge
        if options.get('verbose', True):
            print(f"Warning: Optimization did not converge: {e}")
        popt = p0  # Use initial guess
        pcov = None
        converged = False

    # Construct final parameter dict
    final_params = dict(fixed_params)
    for name, value in zip(fit_params, popt):
        final_params[name] = value

    # Compute predictions and residuals
    predictions = model.forward(data, final_params)
    residuals = y_obs - predictions

    # Compute loss and R²
    loss = np.sum(residuals**2)  # RSS
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    r_squared = 1 - (loss / ss_tot) if ss_tot > 0 else None

    # Construct optimizer info
    optimizer_info = {
        'method': 'scipy.optimize.curve_fit',
        'fitted_parameters': fit_params,
        'fixed_parameters': list(fixed_params.keys())
    }

    return FitResult(
        model=model,
        parameters=final_params,
        data=data,
        predictions=predictions,
        residuals=residuals,
        loss=loss,
        r_squared=r_squared,
        converged=converged,
        iterations=-1,  # scipy doesn't report iterations
        optimizer_info=optimizer_info,
        covariance=pcov,
        fit_options=options or {},
        fit_time=datetime.now()
    )
