---
sidebar_position: 1
---

# API Reference

Complete API reference for PhyTorch modules and functions.

## Unified API

PhyTorch provides a simple, consistent interface for all models:

```python
from phytorch import fit

result = fit(model, data, options=None)
```

### Parameters
- **model**: Any PhyTorch model instance
- **data**: Dictionary mapping variable names to numpy arrays
- **options**: Optional `FitOptions` object for customization

### Returns
- **FitResult**: Object containing fitted parameters, predictions, statistics, and plotting methods

## Core Function

### `fit(model, data, options=None)`

Universal fitting function for all PhyTorch models.

```python
from phytorch import fit
from phytorch.models.generic import Sigmoidal

data = {'x': x_data, 'y': y_data}
result = fit(Sigmoidal(), data)

# Access results
print(result.parameters)  # Fitted parameter values
print(result.r_squared)   # Goodness of fit
predictions = result.predict(new_data)
result.plot()  # Automatic visualization
```

## Available Models

### Generic Models (`phytorch.models.generic`)

General-purpose curve fitting models:

| Model | Description | Required Data |
|-------|-------------|---------------|
| `Linear` | Linear regression y = a*x + b | `x`, `y` |
| `Sigmoidal` | Rational sigmoid curve | `x`, `y` |
| `RectangularHyperbola` | Michaelis-Menten kinetics | `x`, `y` |
| `NonrectangularHyperbola` | Non-rectangular hyperbola | `x`, `y` |
| `Arrhenius` | Temperature response | `x`, `y` |
| `PeakedArrhenius` | Peaked temperature response | `x`, `y` |
| `Gaussian` | Bell-shaped curve | `x`, `y` |
| `Weibull` | Weibull distribution PDF | `x`, `y` |
| `Beta` | Beta distribution | `x`, `y` |

Example:
```python
from phytorch import fit
from phytorch.models.generic import RectangularHyperbola

data = {'x': substrate_conc, 'y': reaction_rate}
result = fit(RectangularHyperbola(), data)
```

### Hydraulics Models (`phytorch.models.hydraulics`)

Plant water relations models:

| Model | Description | Required Data |
|-------|-------------|---------------|
| `Sigmoidal` | Hydraulic vulnerability curve | `x`, `psi` |
| `SJB2018` | Pressure-volume curve (Sack, John, Buckley 2018) | `w`, `psi` |

Example:
```python
from phytorch import fit
from phytorch.models.hydraulics import SJB2018

data = {'w': relative_water_content, 'psi': water_potential}
result = fit(SJB2018(), data)
print(f"Turgor loss point: {result.parameters['w_tlp']:.3f}")
```

### Photosynthesis Models (`phytorch.models.photosynthesis`)

Leaf gas exchange models:

| Model | Description | Required Data |
|-------|-------------|---------------|
| `FvCB` | Farquhar-von Caemmerer-Berry C3 photosynthesis | `Ci`, `Q`, `Tleaf`/`T`, `A` |

Example:
```python
from phytorch import fit
from phytorch.models.photosynthesis import FvCB

data = {
    'Ci': intercellular_co2,
    'Q': light_intensity,
    'Tleaf': leaf_temperature,
    'A': net_photosynthesis
}
result = fit(FvCB(), data)
result.plot()  # Generates comprehensive photosynthesis plots
```

## FitResult Object

The `FitResult` object returned by `fit()` contains:

### Attributes
- `parameters`: Dictionary of fitted parameter values
- `r_squared`: Coefficient of determination
- `rmse`: Root mean squared error
- `data`: Original data used for fitting
- `model`: The fitted model instance

### Methods
- `predict(data)`: Generate predictions for new data
- `plot(save=None, show=True)`: Visualize fit results

```python
# Using FitResult
result = fit(model, data)

# Access fitted parameters
vcmax = result.parameters['Vcmax25']

# Make predictions
predictions = result.predict(new_data)

# Plot results
result.plot(save='my_fit.png')
```

## FitOptions

Customize the fitting process:

```python
from phytorch import fit, FitOptions

options = FitOptions(
    optimizer='scipy',           # 'scipy' or 'adam'
    maxiter=5000,                # Maximum iterations
    bounds={'param': (0, 100)}   # Parameter bounds
)

result = fit(model, data, options)
```

### FitOptions Parameters
- `optimizer`: Optimizer to use ('scipy', 'adam')
- `maxiter`: Maximum number of iterations
- `bounds`: Dictionary of parameter bounds `{param: (lower, upper)}`
- `initial_guess`: Dictionary of initial parameter values

## Model Base Class

All models inherit from the `Model` base class and implement:

```python
class Model:
    def forward(self, data: dict, parameters: dict) -> np.ndarray:
        """Compute model predictions"""
        pass

    def parameter_info(self) -> dict:
        """Return parameter bounds and metadata"""
        pass

    def required_data(self) -> list:
        """Return list of required data fields"""
        pass

    def initial_guess(self, data: dict) -> dict:
        """Generate initial parameter estimates"""
        pass
```

## Quick Reference

### Fitting a Model
```python
from phytorch import fit
from phytorch.models.generic import Sigmoidal

result = fit(Sigmoidal(), {'x': x_data, 'y': y_data})
```

### Plotting Results
```python
result.plot()                    # Show plot
result.plot(save='fit.png')      # Save plot
result.plot(save='fit.png', show=False)  # Save without showing
```

### Making Predictions
```python
new_data = {'x': new_x_values, 'y': new_y_values}
predictions = result.predict(new_data)
```
