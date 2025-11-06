---
sidebar_position: 3
---

# Stomatal Conductance Models

PhyTorch implements several empirical and semi-empirical models of stomatal conductance that link stomatal behavior to environmental conditions and photosynthesis.

## Available Models

PhyTorch provides four widely-used stomatal conductance models:

1. **MED2011** - Medlyn et al. (2011) unified stomatal optimization model
2. **BWB1987** - Ball-Woodrow-Berry (1987) model
3. **BBL1995** - Ball-Berry-Leuning (1995) model
4. **BTA2012** - Buckley-Turnbull-Adams (2012) optimization model

## Basic Usage

All stomatal conductance models follow the same unified API:

```python
from phytorch import fit
from phytorch.models.stomatal import MED2011
import pandas as pd

# Load your gas exchange data
df = pd.read_csv('your_data.csv')

# Prepare data dictionary
data = {
    'A': df['Photo'].values,      # Net CO₂ assimilation (μmol m⁻² s⁻¹)
    'VPD': df['VPDleaf'].values,   # Vapor pressure deficit (kPa)
    'gs': df['Cond'].values,       # Stomatal conductance (mol m⁻² s⁻¹)
    'Ca': df['CO2_r'].values       # Atmospheric CO₂ (ppm), optional
}

# Fit the model (that's it!)
result = fit(MED2011(), data)

# View fitted parameters
print(f"gs0: {result.parameters['gs0']:.4f} mol/m²/s")
print(f"g1: {result.parameters['g1']:.2f}")
print(f"R² = {result.r_squared:.4f}")

# Generate plots
result.plot()
```

## Medlyn Model (MED2011)

The Medlyn model (also called the unified stomatal optimization or USO model) relates stomatal conductance to assimilation rate and vapor pressure deficit:

$$
g_s = g_{s0} + 1.6 \left(1 + \frac{{g_1}}{{\sqrt{{VPD}}}}\right) \frac{{A}}{{C_a}}
$$

where:
- $g_s$ = Stomatal conductance to water vapor (mol m⁻² s⁻¹)
- $g_{s0}$ = Minimum/residual conductance (mol m⁻² s⁻¹)
- $g_1$ = Slope parameter (dimensionless)
- $VPD$ = Vapor pressure deficit (kPa)
- $A$ = Net CO₂ assimilation rate (μmol m⁻² s⁻¹)
- $C_a$ = Atmospheric CO₂ concentration (ppm)
- 1.6 = Ratio of water to CO₂ diffusivity

### Usage

```python
from phytorch import fit
from phytorch.models.stomatal import MED2011

# Fit the model
model = MED2011()
result = fit(model, data)

print(f"gs0 = {result.parameters['gs0']:.4f} mol/m²/s")
print(f"g1 = {result.parameters['g1']:.2f}")
```

### Parameters

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| `gs0` | Minimum conductance | 0.0-0.1 | mol/m²/s |
| `g1` | Slope parameter | 0.5-15.0 | dimensionless |

## Ball-Woodrow-Berry Model (BWB1987)

The BWB model relates stomatal conductance to assimilation, relative humidity, and CO₂:

$$
g_s = g_{s0} + g_1 \frac{{A \cdot RH}}{{C_s}}
$$

where:
- $g_s$ = Stomatal conductance to water vapor (mol m⁻² s⁻¹)
- $g_{s0}$ = Minimum conductance (mol m⁻² s⁻¹)
- $g_1$ = Slope parameter (dimensionless)
- $A$ = Net CO₂ assimilation rate (μmol m⁻² s⁻¹)
- $RH$ = Relative humidity (fraction, 0-1)
- $C_s$ = CO₂ concentration at leaf surface (ppm)

### Usage

```python
from phytorch import fit
from phytorch.models.stomatal import BWB1987

# Prepare data (requires relative humidity)
data = {
    'A': df['Photo'].values,
    'RH': df['RH_s'].values / 100,  # Convert % to fraction
    'Cs': df['CO2_s'].values,
    'gs': df['Cond'].values
}

# Fit the model
result = fit(BWB1987(), data)

print(f"gs0 = {result.parameters['gs0']:.4f} mol/m²/s")
print(f"g1 = {result.parameters['g1']:.2f}")
```

### Parameters

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| `gs0` | Minimum conductance | 0.0-0.1 | mol/m²/s |
| `g1` | Slope parameter | 5-15 | dimensionless |

## Ball-Berry-Leuning Model (BBL1995)

The BBL model is an extension of the BWB model that uses VPD instead of relative humidity and includes a CO₂ compensation point:

$$
g_s = g_{s0} + \frac{{a_1 \cdot A}}{{(C_a - \Gamma)(1 + VPD/D_0)}}
$$

where:
- $g_s$ = Stomatal conductance to water vapor (mol m⁻² s⁻¹)
- $g_{s0}$ = Minimum conductance (mol m⁻² s⁻¹)
- $a_1$ = Slope parameter (dimensionless)
- $A$ = Net CO₂ assimilation rate (μmol m⁻² s⁻¹)
- $C_a$ = Atmospheric CO₂ concentration (ppm)
- $\Gamma$ = CO₂ compensation point (ppm)
- $VPD$ = Vapor pressure deficit (kPa)
- $D_0$ = VPD sensitivity parameter (kPa)

### Usage

```python
from phytorch import fit
from phytorch.models.stomatal import BBL1995

# Prepare data
data = {
    'A': df['Photo'].values,
    'VPD': df['VPDleaf'].values,
    'Ca': df['CO2_r'].values,
    'gs': df['Cond'].values
}

# Fit the model
result = fit(BBL1995(), data)

print(f"gs0 = {result.parameters['gs0']:.4f} mol/m²/s")
print(f"a1 = {result.parameters['a1']:.2f}")
print(f"D0 = {result.parameters['D0']:.2f} kPa")
```

### Parameters

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| `gs0` | Minimum conductance | 0.0-0.1 | mol/m²/s |
| `a1` | Slope parameter | 5-15 | dimensionless |
| `D0` | VPD sensitivity | 0.5-3.0 | kPa |

## Buckley-Turnbull-Adams Model (BTA2012)

The BTA model (Model 4 from Buckley et al. 2012) relates stomatal conductance to irradiance and vapor pressure deficit using lumped parameters:

$$
g_s = \frac{{E_m(Q + i_0)}}{{k + bQ + (Q + i_0)D_s}}
$$

where:
- $g_s$ = Stomatal conductance to water vapor (mol m⁻² s⁻¹)
- $E_m$ = Maximum leaf transpiration rate (mmol m⁻² s⁻¹)
- $Q$ = Irradiance/PPFD (μmol m⁻² s⁻¹)
- $i_0$ = Dark respiration parameter (μmol m⁻² s⁻¹)
- $k$ = Lumped parameter $K_1/\chi\phi$ (μmol m⁻² s⁻¹ mmol mol⁻¹)
- $b$ = Lumped parameter $K_1/\chi\alpha_0$ (mmol mol⁻¹)
- $D_s$ = Leaf surface vapor pressure saturation deficit (mmol mol⁻¹)

The lumped parameters are defined as:
- $E_m = K_1(\psi_{soil} + \pi_c)$ where $K_1$ is leaf specific hydraulic conductance, $\psi_{soil}$ is soil water potential, and $\pi_c$ is epidermal osmotic pressure
- $k = K_1/\chi\phi$ where $\chi$ is the turgor to conductance scalar and $\phi$ is the initial slope of $g_s$ response to irradiance
- $b = K_1/\chi\alpha_0$ where $\alpha_0$ is α in darkness divided by $\phi$

### Usage

```python
from phytorch import fit
from phytorch.models.stomatal import BTA2012

# Prepare data
data = {
    'Q': df['PARi'].values,      # Irradiance (μmol m⁻² s⁻¹)
    'Ds': df['VPDleaf'].values,  # Leaf surface VPD (mmol mol⁻¹)
    'gs': df['Cond'].values      # Stomatal conductance (mol m⁻² s⁻¹)
}

# Fit the model
result = fit(BTA2012(), data)

print(f"Em = {result.parameters['Em']:.3f} mmol/m²/s")
print(f"i0 = {result.parameters['i0']:.2f} μmol/m²/s")
print(f"k = {result.parameters['k']:.2f}")
print(f"b = {result.parameters['b']:.3f} mmol/mol")
```

### Parameters

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| `Em` | Maximum transpiration rate | 0.1-50 | mmol/m²/s |
| `i0` | Dark respiration parameter | 0-300 | μmol/m²/s |
| `k` | Lumped parameter K₁/χφ | 0-1×10⁶ (typically ~1×10⁴) | μmol m⁻² s⁻¹ mmol mol⁻¹ |
| `b` | Lumped parameter K₁/χα₀ | 0-100 (typically ~6.7) | mmol/mol |

## Advanced Features

### Custom Parameter Constraints

```python
from phytorch import fit
from phytorch.models.stomatal import MED2011

# Define custom parameter bounds
options = {
    'bounds': {
        'gs0': (0.0, 0.05),
        'g1': (2.0, 8.0)
    }
}

result = fit(MED2011(), data, options)
```

### Making Predictions

```python
# Fit the model
result = fit(MED2011(), training_data)

# Make predictions on new data
new_data = {
    'A': np.array([10, 15, 20, 25]),
    'VPD': np.array([1.0, 1.5, 2.0, 2.5]),
    'Ca': 400
}

predictions = result.predict(new_data)
print(f"Predicted gs: {predictions}")
```

## Comparing Models

Different models may be appropriate for different species or environmental conditions. Note that BTA2012 requires different input data (irradiance and Ds) compared to the other models:

```python
from phytorch import fit
from phytorch.models.stomatal import MED2011, BWB1987, BBL1995, BTA2012

# Prepare data for MED2011, BWB1987, and BBL1995
data_common = {
    'A': df['Photo'].values,
    'VPD': df['VPDleaf'].values,
    'Ca': df['CO2_r'].values,
    'gs': df['Cond'].values
}

# Prepare data for BTA2012 (requires different inputs)
data_bta = {
    'Q': df['PARi'].values,
    'Ds': df['VPDleaf'].values,  # or convert VPD to mmol mol⁻¹
    'gs': df['Cond'].values
}

# Fit models with common data requirements
models_common = {
    'Medlyn': MED2011(),
    'BBL': BBL1995()
}

results = {}
for name, model in models_common.items():
    results[name] = fit(model, data_common)
    print(f"{name}: R² = {results[name].r_squared:.4f}")

# Fit BTA model separately with its specific data
results['BTA'] = fit(BTA2012(), data_bta)
print(f"BTA: R² = {results['BTA'].r_squared:.4f}")
```

## References

- Medlyn, B. E., et al. (2011). Reconciling the optimal and empirical approaches to modelling stomatal conductance. *Global Change Biology*, 17(6), 2134-2144.
- Ball, J. T., Woodrow, I. E., & Berry, J. A. (1987). A model predicting stomatal conductance and its contribution to the control of photosynthesis under different environmental conditions. *Progress in Photosynthesis Research*, 4, 221-224.
- Leuning, R. (1995). A critical appraisal of a combined stomatal-photosynthesis model for C3 plants. *Plant, Cell & Environment*, 18(4), 339-355.
- Buckley, T. N., Turnbull, T. L., & Adams, M. A. (2012). Simple models for stomatal conductance derived from a process model: cross-validation against sap flux data. *Plant, Cell & Environment*, 35(9), 1647-1662.
