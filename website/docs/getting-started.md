---
sidebar_position: 3
---

# Getting Started

This guide will walk you through fitting your first photosynthesis model using PhyTorch.

## Basic Photosynthesis Model Fitting

Let's fit the Farquhar-von Caemmerer-Berry (FvCB) model to LI-COR A-Ci curve data.

### 1. Import Required Modules

```python
from phytorch import *
import pandas as pd
import torch
import matplotlib.pyplot as plt
```

### 2. Load Your Data

PhyTorch works with pandas DataFrames containing LI-COR gas exchange data:

```python
# Load your LI-COR data (CSV format)
df = pd.read_csv('your_licor_data.csv')

# Initialize LI-COR data object with preprocessing
lcd = fvcb.initLicordata(df, preprocess=True)
```

The data should contain columns for:
- `Ci` - Intercellular CO2 concentration (μmol/mol)
- `Photo` - Net photosynthesis rate (μmol/m²/s)
- `Tleaf` - Leaf temperature (°C)
- `PARi` - Incident photosynthetically active radiation (μmol/m²/s)

### 3. Initialize and Fit the FvCB Model

```python
# Initialize FvCB model
# LightResp_type: 1 = non-rectangular hyperbola, 2 = rectangular hyperbola
# TempResp_type: 1 = Arrhenius, 2 = peaked Arrhenius
fvcbm = fvcb.model(lcd, LightResp_type=2, TempResp_type=2)

# Fit the model
fitresult = fvcb.fit(
    fvcbm,
    learn_rate=0.08,      # Learning rate for optimization
    maxiteration=20000    # Maximum iterations
)
```

### 4. View Results

```python
# Print fitted parameters
print("Fitted Parameters:")
print(f"Vcmax25: {fitresult.params['Vcmax25']:.2f} μmol/m²/s")
print(f"Jmax25: {fitresult.params['Jmax25']:.2f} μmol/m²/s")
print(f"Rd25: {fitresult.params['Rd25']:.2f} μmol/m²/s")
print(f"Tp25: {fitresult.params['Tp25']:.2f} μmol/m²/s")

# Get fitted values
fitted_values = fvcbm.predict(lcd)
```

### 5. Visualize A-Ci Curves

```python
# Plot observed vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(lcd.Ci, lcd.Photo, alpha=0.6, label='Observed')
plt.scatter(lcd.Ci, fitted_values, alpha=0.6, label='Fitted')
plt.xlabel('Ci (μmol/mol)')
plt.ylabel('A (μmol/m²/s)')
plt.legend()
plt.title('FvCB Model Fit')
plt.grid(True, alpha=0.3)
plt.show()
```

## Stomatal Conductance Models

PhyTorch supports multiple stomatal conductance models including Medlyn (MED), Ball-Woodrow-Berry (BWB), and Buckley-Mott-Farquhar (BMF).

```python
from phytorch import *
import pandas as pd

# Load your LI-COR data
df = pd.read_csv('your_licor_data.csv')
lcd = stomatal.initLicordata(df, preprocess=True)

# Fit Medlyn model
medlyn_model = stomatal.model(lcd, model_type='MED')
medlyn_result = stomatal.fit(medlyn_model, learn_rate=0.01, maxiteration=10000)

# View fitted parameters
print(f"g0: {medlyn_result.params['g0']:.4f} mol/m²/s")
print(f"g1: {medlyn_result.params['g1']:.2f}")

# Fit Ball-Woodrow-Berry model
bwb_model = stomatal.model(lcd, model_type='BWB')
bwb_result = stomatal.fit(bwb_model, learn_rate=0.01, maxiteration=10000)
```

## Temperature Response Options

PhyTorch supports different temperature response functions:

### TempResp_type Options

- **Type 1**: Arrhenius function
  ```
  f(T) = f25 * exp[Ha/R * (1/298.15 - 1/Tk)]
  ```

- **Type 2**: Peaked Arrhenius function (includes deactivation)
  ```
  f(T) = f25 * exp[Ha/R * (1/298.15 - 1/Tk)] * [1 + exp(S*298.15 - Hd)/(R*298.15)] / [1 + exp(S*Tk - Hd)/(R*Tk)]
  ```

```python
# Use peaked Arrhenius for temperature-sensitive parameters
fvcbm = fvcb.model(lcd, LightResp_type=2, TempResp_type=2)
```

## Next Steps

- Explore different [photosynthesis models](./models/photosynthesis.md)
- Learn about [stomatal conductance models](./models/stomatal-conductance.md)
- Check out the [API Reference](./api/index.md) for detailed documentation
- See [examples and tutorials](./tutorials/index.md) for more advanced use cases
