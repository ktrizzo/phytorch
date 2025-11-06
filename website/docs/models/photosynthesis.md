---
sidebar_position: 2
---

# Photosynthesis Models

PhyTorch implements the Farquhar-von Caemmerer-Berry (FvCB) model for C3 photosynthesis, one of the most widely used biochemical models of leaf photosynthesis.

## FvCB Model

The FvCB model calculates net CO2 assimilation rate ($A$) as the minimum of three potentially limiting rates:

$$
A = \min(W_c, W_j, W_p) \left(1 - \frac{{\Gamma^*}}{{C_i}}\right) - R_d
$$

where:
- **Rubisco-limited rate ($W_c$)**: Limited by the maximum carboxylation rate ($V_{{cmax}}$)
- **RuBP regeneration-limited rate ($W_j$)**: Limited by electron transport capacity ($J_{{max}}$)
- **Triose phosphate utilization-limited rate ($W_p$)**: Limited by the capacity to use photosynthetic products
- **$\Gamma^*$**: CO2 compensation point in the absence of day respiration
- **$C_i$**: Intercellular CO2 concentration
- **$R_d$**: Day respiration

### Rubisco-Limited Rate

$$
W_c = \frac{{V_{{cmax}} \cdot C_i}}{{C_i + K_c(1 + O/K_o)}}
$$

### RuBP Regeneration-Limited Rate

$$
W_j = \frac{{J \cdot C_i}}{{4C_i + 8\Gamma^*}}
$$

where $J$ is the electron transport rate, calculated from light response

### Triose Phosphate Utilization-Limited Rate

$$
W_p = \frac{{3 \cdot TPU \cdot C_i}}{{C_i - \Gamma^*(1 + 3\alpha_G)}}
$$

where:
- $TPU$ is the triose phosphate utilization rate
- $\alpha_G$ is the fraction of glycolate carbon not returned to the chloroplast (typically 0-0.5)

### Basic Usage

PhyTorch's unified API makes fitting photosynthesis models simple and consistent:

```python
from phytorch import fit
from phytorch.models.photosynthesis import FvCB
import pandas as pd

# Load LI-COR A-Ci curve data
df = pd.read_csv('your_aci_data.csv')

# Prepare data dictionary
data = {
    'Ci': df['Ci'].values,       # Intercellular CO2 (μmol/mol)
    'Q': df['PARi'].values,       # Light intensity (μmol/m²/s)
    'Tleaf': df['Tleaf'].values,  # Leaf temperature (°C)
    'A': df['Photo'].values       # Net photosynthesis (μmol/m²/s)
}

# Fit the FvCB model (that's it!)
result = fit(FvCB(), data)

# View fitted parameters
print(f"Vcmax25: {result.parameters['Vcmax25']:.2f} μmol/m²/s")
print(f"Jmax25: {result.parameters['Jmax25']:.2f} μmol/m²/s")
print(f"Rd25: {result.parameters['Rd25']:.2f} μmol/m²/s")
print(f"R² = {result.r_squared:.4f}")

# Generate comprehensive plots
# Includes: 1:1, A vs Ci, A vs Q, A vs T, and 3D surfaces
result.plot()

# Save the plot
result.plot(save='fvcb_fit.png')
```

### Key Parameters

PhyTorch fits the following parameters at 25°C:

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| `Vcmax25` | Maximum carboxylation rate at 25°C | 50-150 | μmol/m²/s |
| `Jmax25` | Maximum electron transport rate at 25°C | 100-250 | μmol/m²/s |
| `Rd25` | Day respiration at 25°C | 0.5-2.0 | μmol/m²/s |
| `Tp25` | Triose phosphate utilization at 25°C | 5-15 | μmol/m²/s |

### Biochemical Constants

| Parameter | Description | Value at 25°C | Units |
|-----------|-------------|---------------|-------|
| `Kc` | Michaelis constant for CO2 | 260 | μmol/mol |
| `Ko` | Michaelis constant for O2 | 179 | mmol/mol |
| `Γ*` | CO2 compensation point | ~40 | μmol/mol |

### Temperature Response

PhyTorch supports two temperature response types:

#### Type 1: Arrhenius Function

```python
# Simple Arrhenius temperature response
fvcbm = fvcb.model(lcd, LightResp_type=2, TempResp_type=1)
```

The Arrhenius function:

$$
k = k_{{25}} \exp\left[\frac{{\Delta H_a}}{{R}}\left(\frac{{1}}{{298}}-\frac{{1}}{{T_{{leaf}}}}\right)\right]
$$

| Parameter | Description | Value/Range | Units |
|-----------|-------------|-------------|-------|
| $k$ | Parameter value (Vcmax, Jmax, or TPU) | Variable | μmol/m²/s |
| $k_{{25}}$ | Parameter value at 25°C | Variable | μmol/m²/s |
| $\Delta H_a$ | Activation energy | 50,000-100,000 | J/mol |
| $R$ | Universal gas constant | 8.314 | J/mol/K |
| $T_{{leaf}}$ | Leaf temperature | 273-323 (0-50°C) | K |

#### Type 2: Peaked Arrhenius Function

```python
# Peaked Arrhenius (includes high-temperature deactivation)
fvcbm = fvcb.model(lcd, LightResp_type=2, TempResp_type=2)
```

The peaked Arrhenius function:

$$
k = k_{{25}} \exp\left[\frac{{\Delta H_a}}{{R}} \left(\frac{{1}}{{298}}-\frac{{1}}{{T_{{leaf}}}}\right)\right] \frac{{f(298)}}{{f(T_{{leaf}})}}
$$

where:

$$
f(T) = 1+\exp \left[\frac{{\Delta H_d}}{{R}}\left(\frac{{1}}{{T_{{opt}}}}-\frac{{1}}{{T}} \right)-\ln \left(\frac{{\Delta H_d}}{{\Delta H_a}}-1 \right) \right]
$$

| Parameter | Description | Value/Range | Units |
|-----------|-------------|-------------|-------|
| $k$ | Parameter value (Vcmax, Jmax, or TPU) | Variable | μmol/m²/s |
| $k_{{25}}$ | Parameter value at 25°C | Variable | μmol/m²/s |
| $\Delta H_a$ | Activation energy | 50,000-100,000 | J/mol |
| $\Delta H_d$ | Deactivation energy | 150,000-250,000 | J/mol |
| $T_{{opt}}$ | Optimal temperature | 298-313 (25-40°C) | K |
| $R$ | Universal gas constant | 8.314 | J/mol/K |
| $T_{{leaf}}$ | Leaf temperature | 273-323 (0-50°C) | K |

### Light Response

PhyTorch supports three light response types for electron transport rate ($J$):

#### Type 0: No Light Dependence

```python
fvcbm = fvcb.model(lcd, LightResp_type=0, TempResp_type=2)
```

$$
J = J_{{max}}
$$

No additional parameters are fitted. Electron transport is simply equal to $J_{{max}}$.

#### Type 1: Rectangular Hyperbola

```python
fvcbm = fvcb.model(lcd, LightResp_type=1, TempResp_type=2)
```

$$
J = \frac{{\alpha Q J_{{max}}}}{{\alpha Q + J_{{max}}}}
$$

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| $J_{{max}}$ | Maximum electron transport rate | 100-250 | μmol/m²/s |
| $\alpha$ | Light use efficiency | 0.2-0.4 | mol e⁻/mol photons |
| $Q$ | Photosynthetic photon flux density (PPFD) | 0-2500 | μmol/m²/s |

#### Type 2: Non-Rectangular Hyperbola

```python
fvcbm = fvcb.model(lcd, LightResp_type=2, TempResp_type=2)
```

$$
J = \frac{{\alpha Q + J_{{max}} - \sqrt{{(\alpha Q + J_{{max}})^2 - 4 \theta \alpha Q J_{{max}}}}}}{{2 \theta}}
$$

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| $J_{{max}}$ | Maximum electron transport rate | 100-250 | μmol/m²/s |
| $\alpha$ | Light use efficiency | 0.2-0.4 | mol e⁻/mol photons |
| $\theta$ | Curvature parameter | 0.7-0.9 | dimensionless |
| $Q$ | Photosynthetic photon flux density (PPFD) | 0-2500 | μmol/m²/s |

### A-Ci Curves

Fitting A-Ci (assimilation vs. intercellular CO2) curves:

```python
from phytorch import fit
from phytorch.models.photosynthesis import FvCB
import numpy as np

# Your A-Ci curve data
data = {
    'Ci': np.array([50, 100, 200, 400, 600, 800, 1000]),
    'Q': np.full(7, 1500),  # Constant light at 1500 μmol/m²/s
    'Tleaf': np.full(7, 25),  # Constant temperature at 25°C
    'A': np.array([5.2, 10.5, 18.3, 24.1, 26.8, 28.2, 29.1])
}

# Fit the model
result = fit(FvCB(), data)

# View results
print(f"Vcmax25: {result.parameters['Vcmax25']:.2f} μmol/m²/s")
print(f"Jmax25: {result.parameters['Jmax25']:.2f} μmol/m²/s")
print(f"Rd25: {result.parameters['Rd25']:.2f} μmol/m²/s")

# Plot comprehensive results
result.plot()
```

## Advanced Features

### Custom Parameter Constraints

```python
from phytorch import fit, FitOptions
from phytorch.models.photosynthesis import FvCB

# Define custom parameter bounds
options = FitOptions(
    bounds={
        'Vcmax25': (20, 200),
        'Jmax25': (40, 400),
        'Rd25': (0, 5)
    }
)

result = fit(FvCB(), data, options)
```

### Custom Initial Guesses

```python
from phytorch import fit, FitOptions

# Provide initial parameter estimates
options = FitOptions(
    initial_guess={
        'Vcmax25': 100,
        'Jmax25': 180,
        'Rd25': 1.5
    }
)

result = fit(FvCB(), data, options)
```

### Making Predictions

```python
# Fit the model
result = fit(FvCB(), training_data)

# Make predictions on new data
new_data = {
    'Ci': np.linspace(50, 1000, 100),
    'Q': np.full(100, 1500),
    'Tleaf': np.full(100, 25)
}

predictions = result.predict(new_data)
```

## References

- Farquhar, G. D., von Caemmerer, S., & Berry, J. A. (1980). A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species. *Planta*, 149(1), 78-90.
- Sharkey, T. D., et al. (2007). Fitting photosynthetic carbon dioxide response curves for C3 leaves. *Plant, Cell & Environment*, 30(9), 1035-1040.
