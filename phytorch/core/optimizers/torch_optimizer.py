"""PyTorch-based optimizer for complex coupled models like FvCB."""

from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
from datetime import datetime

from phytorch.core.result import FitResult


def fit_with_torch(
    model: nn.Module,
    data: Dict,
    options: Optional[Dict] = None
) -> FitResult:
    """Fit model using PyTorch gradient descent.

    Args:
        model: PyTorch nn.Module with FvCB-like interface
        data: Input data dict
        options: Fitting options
            - learn_rate: Learning rate (default: 0.6)
            - max_iterations: Maximum iterations (default: 20000)
            - min_loss: Stop if loss below this (default: 3.0)
            - device: 'cpu' or 'cuda' (default: 'cpu')
            - verbose: Print progress (default: True)
            - scheduler_step: LR scheduler step size (default: 5000)
            - scheduler_gamma: LR scheduler decay (default: 0.8)

    Returns:
        FitResult with fitted parameters and diagnostics
    """
    options = options or {}

    # Get options
    learn_rate = options.get('learn_rate', 0.6)
    max_iterations = options.get('max_iterations', 20000)
    min_loss = options.get('min_loss', 3.0)
    device_name = options.get('device', 'cpu')
    verbose = options.get('verbose', True)
    scheduler_step = options.get('scheduler_step', 5000)
    scheduler_gamma = options.get('scheduler_gamma', 0.8)

    # Setup device
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        model.to(device)
        use_amp = True
        scaler = GradScaler()
    else:
        device = torch.device('cpu')
        use_amp = False
        scaler = None

    # Initialize model with data (for models that need it)
    # This ensures parameters are created before optimizer
    if hasattr(model, '_prepare_data'):
        model._prepare_data(data)

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    # Track losses
    if device.type == 'cuda':
        loss_history = torch.tensor([]).to(device)
    else:
        loss_history = torch.tensor([])

    best_loss = 1e12
    best_weights = model.state_dict()
    best_iter = 0

    start_time = time.time()

    # Training loop
    for iter in range(max_iterations):
        optimizer.zero_grad()

        # Forward pass
        if use_amp:
            with autocast():
                loss = model.compute_loss(data)
        else:
            loss = model.compute_loss(data)

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        # Track loss
        loss_history = torch.cat((loss_history, loss.unsqueeze(0)), dim=0)

        # Print progress
        if (iter + 1) % 200 == 0 and verbose:
            print(f'Loss at iter {iter}: {loss.item():.4f}')

        # Track best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = model.state_dict()
            best_iter = iter

        # Early stopping
        if loss.item() < min_loss:
            if verbose:
                print(f'Fitting stopped at iter {iter}')
            break

    end_time = time.time()
    elapsed_time = end_time - start_time

    if verbose:
        print(f'Best loss at iter {best_iter}: {best_loss:.4f}')
        print(f'Fitting time: {elapsed_time:.4f} seconds')

    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()

    # Get final predictions and parameters
    with torch.no_grad():
        predictions = model.forward(data)
        if isinstance(predictions, tuple):
            # FvCB returns (A, Ac, Aj, Ap) - use A for primary prediction
            predictions = predictions[0]

        # Convert to numpy
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()

    # Extract parameters from model
    parameters = {}
    for name, param in model.named_parameters():
        parameters[name] = param.detach().cpu().item() if param.numel() == 1 else param.detach().cpu().numpy()

    # Get observed data
    # For models with custom data structures (like FvCB), use get_observed_data()
    if hasattr(model, 'get_observed_data'):
        y_obs = model.get_observed_data()
    else:
        required_fields = model.required_data()
        y_field = required_fields[-1]
        y_obs = np.asarray(data[y_field])

    # Compute residuals and metrics
    residuals = y_obs - predictions
    loss_final = np.sum(residuals**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    r_squared = 1 - (loss_final / ss_tot) if ss_tot > 0 else None

    # Construct optimizer info
    optimizer_info = {
        'method': 'torch.optim.Adam',
        'learn_rate': learn_rate,
        'iterations': best_iter,
        'final_loss': best_loss,
        'elapsed_time': elapsed_time
    }

    return FitResult(
        model=model,
        parameters=parameters,
        data=data,
        predictions=predictions,
        residuals=residuals,
        loss=loss_final,
        r_squared=r_squared,
        converged=True,  # Assume converged if we finished
        iterations=best_iter,
        optimizer_info=optimizer_info,
        covariance=None,  # Not available for gradient descent
        fit_options=options or {},
        fit_time=datetime.now()
    )
