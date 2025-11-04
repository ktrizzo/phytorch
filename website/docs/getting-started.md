---
sidebar_position: 3
---

# Getting Started

This guide will walk you through fitting your first model using PhyTorch's unified API.

## PhyTorch's Unified API

PhyTorch provides a simple, consistent interface for fitting models across all domains:

```python
from phytorch import fit

# Universal API for all models
result = fit(model, data, options)
```

All models follow the same pattern:
1. **Import your model** from the appropriate module
2. **Prepare your data** as a dictionary with variable names
3. **Fit the model** with a single function call
4. **Visualize results** with built-in plotting

## Quick Start: Fitting a Simple Model

Let's start with a simple curve-fitting example:

```python
from phytorch import fit
from phytorch.models.generic import Sigmoidal
import numpy as np

# Prepare your data
data = {
    'x': np.array([-3, -2, -1, 0, 1, 2, 3]),
    'y': np.array([0.5, 1.2, 2.5, 5.0, 7.5, 9.0, 9.8])
}

# Fit the model (that's it!)
result = fit(Sigmoidal(), data)

# View fitted parameters
print(result.parameters)

# Plot the results
result.plot()
```

## Example: Photosynthesis Model

Fitting a photosynthesis model is just as simple:

```python
from phytorch import fit
from phytorch.models.photosynthesis import FvCB
import pandas as pd

# Load your A-Ci curve data
df = pd.read_csv('aci_data.csv')

# Prepare data dictionary
data = {
    'Ci': df['Ci'].values,      # Intercellular CO2 (μmol/mol)
    'Q': df['PARi'].values,      # Light intensity (μmol/m²/s)
    'Tleaf': df['Tleaf'].values, # Leaf temperature (°C)
    'A': df['Photo'].values      # Net photosynthesis (μmol/m²/s)
}

# Fit the FvCB model
result = fit(FvCB(), data)

# View fitted parameters
print(f"Vcmax25: {result.parameters['Vcmax25']:.2f} μmol/m²/s")
print(f"Jmax25: {result.parameters['Jmax25']:.2f} μmol/m²/s")
print(f"R² = {result.r_squared:.4f}")

# Generate comprehensive plots
# For photosynthesis: 1:1, A vs Ci, A vs Q, A vs T, plus 3D surfaces
result.plot()
```

## Example: Hydraulic Model

Fitting pressure-volume curves uses the same unified API:

```python
from phytorch import fit
from phytorch.models.hydraulics import SJB2018
import numpy as np

# Prepare P-V curve data
data = {
    'w': np.array([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]),
    'psi': np.array([-2.8, -2.1, -1.5, -1.0, -0.6, -0.3, -0.1])
}

# Fit the SJB2018 model
result = fit(SJB2018(), data)

# View fitted parameters
print(f"pi_o: {result.parameters['pi_o']:.2f} MPa")
print(f"w_tlp: {result.parameters['w_tlp']:.2f}")
print(f"epsilon: {result.parameters['epsilon']:.2f}")

# Plot predicted vs observed and model fit
result.plot()
```

## Available Model Categories

PhyTorch provides a comprehensive library of models across multiple domains:

### Generic Models
Perfect for general curve fitting:
- `Linear` - Linear regression
- `Sigmoidal` - Rational sigmoid curves
- `RectangularHyperbola` - Michaelis-Menten kinetics
- `NonrectangularHyperbola` - Non-rectangular hyperbola
- `Arrhenius` - Temperature response
- `PeakedArrhenius` - Temperature response with high-temp deactivation
- `Gaussian` - Bell-shaped curves
- `Weibull` - Weibull distributions
- `Beta` - Beta distributions

### Hydraulics Models
For plant water relations:
- `Sigmoidal` - Simple hydraulic vulnerability curves
- `SJB2018` - Pressure-volume curves (Sack, John, Buckley 2018)

### Photosynthesis Models
For leaf gas exchange:
- `FvCB` - Farquhar-von Caemmerer-Berry C3 photosynthesis

All models use the same simple API pattern!

## Built-in Plotting

All fit results include automatic plotting functionality that adapts to the model type:

### 1D Models (Single Input)
```python
result = fit(Sigmoidal(), data)
result.plot()  # Shows: Predicted vs Observed + Model Fit curve
```

### Multi-dimensional Models
```python
result = fit(SomeMultiDModel(), data)
result.plot()  # Shows: 1:1 plot + response curves for each variable
```

### Photosynthesis Models (Special Treatment)
```python
result = fit(FvCB(), data)
result.plot()  # Shows: 1:1, A vs Ci, A vs Q, A vs T,
               # plus 3D surfaces (Ci-Q-A and Ci-T-A)
```

Save plots directly:
```python
result.plot(save='my_fit.png', show=False)
```

## Customization Options

### Custom Parameter Bounds
```python
from phytorch import fit, FitOptions

options = FitOptions(
    bounds={
        'ymax': (0, 100),
        'x50': (-5, 5)
    }
)

result = fit(Sigmoidal(), data, options)
```

### Custom Optimizer Settings
```python
options = FitOptions(
    optimizer='scipy',
    maxiter=5000
)

result = fit(model, data, options)
```

## Why PhyTorch?

### Simple and Consistent
One unified API for all models - no need to learn different interfaces for different model types.

### Comprehensive
From simple linear regression to complex photosynthesis models, all in one toolkit.

### Automatic Visualization
Built-in plotting adapts to your model type, making it easy to verify fit quality.

## Next Steps

- Explore different [photosynthesis models](./models/photosynthesis.md)
- Check out the [API Reference](./api/index.md) for detailed documentation
- See the complete [model library](./api/index.md#available-models)
