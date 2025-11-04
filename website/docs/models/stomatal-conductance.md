---
sidebar_position: 3
---

# Stomatal Conductance Models

PhyTorch implements several empirical and semi-empirical models of stomatal conductance that link stomatal behavior to environmental conditions and photosynthesis.

## Medlyn Model (MED)

The Medlyn model (also called the USO model) relates stomatal conductance to assimilation rate and vapor pressure deficit:

$$
g_s = g_0 + 1.6 \left(1 + \frac{{g_1}}{{\sqrt{{D}}}}\right) \frac{{A}}{{C_a}}
$$

where:
- $g_s$ = Stomatal conductance to water vapor (mol m⁻² s⁻¹)
- $g_0$ = Residual conductance (mol m⁻² s⁻¹)
- $g_1$ = Slope parameter (√kPa)
- $D$ = Vapor pressure deficit (kPa)
- $A$ = Net CO₂ assimilation rate (μmol m⁻² s⁻¹)
- $C_a$ = Atmospheric CO₂ concentration (μmol mol⁻¹)

### Usage

```python
from phytorch import *
import pandas as pd

# Load LI-COR data
df = pd.read_csv('your_gs_data.csv')
lcd = stomatal.initLicordata(df, preprocess=True)

# Initialize Medlyn model
med_model = stomatal.model(lcd, model_type='MED')

# Fit the model
fitresult = stomatal.fit(
    med_model,
    learn_rate=0.01,
    maxiteration=10000
)

# View fitted parameters
print(f"g0 = {fitresult.params['g0']:.4f} mol/m²/s")
print(f"g1 = {fitresult.params['g1']:.2f}")
```

### Parameters

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| `g0` | Residual conductance | 0.0-0.05 | mol/m²/s |
| `g1` | Slope parameter | 2-8 | √kPa |

## Ball-Woodrow-Berry Model (BWB)

The BWB model relates stomatal conductance to assimilation, relative humidity, and CO2:

$$
g_s = g_0 + g_1 \frac{{A \cdot RH}}{{C_s}}
$$

where:
- $g_s$ = Stomatal conductance to water vapor (mol m⁻² s⁻¹)
- $g_0$ = Residual conductance (mol m⁻² s⁻¹)
- $g_1$ = Slope parameter (dimensionless)
- $A$ = Net CO₂ assimilation rate (μmol m⁻² s⁻¹)
- $RH$ = Relative humidity (0-1)
- $C_s$ = CO₂ concentration at leaf surface (μmol mol⁻¹)

### Usage

```python
from phytorch import *
import pandas as pd

# Load LI-COR data
df = pd.read_csv('your_gs_data.csv')
lcd = stomatal.initLicordata(df, preprocess=True)

# Initialize BWB model
bwb_model = stomatal.model(lcd, model_type='BWB')

# Fit the model
fitresult = stomatal.fit(
    bwb_model,
    learn_rate=0.01,
    maxiteration=10000
)

# View fitted parameters
print(f"g0 = {fitresult.params['g0']:.4f} mol/m²/s")
print(f"g1 = {fitresult.params['g1']:.2f}")
```

### Parameters

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| `g0` | Residual conductance | 0.0-0.05 | mol/m²/s |
| `g1` | Slope parameter | 5-15 | dimensionless |

## Buckley-Mott-Farquhar Model (BMF)

The BMF model is a more mechanistic approach based on optimization theory.

```python
from phytorch import *
import pandas as pd

# Load LI-COR data
df = pd.read_csv('your_gs_data.csv')
lcd = stomatal.initLicordata(df, preprocess=True)

# Initialize BMF model
bmf_model = stomatal.model(lcd, model_type='BMF')

# Fit the model
fitresult = stomatal.fit(
    bmf_model,
    learn_rate=0.01,
    maxiteration=10000
)
```

## Ball-Berry-Leuning Model (BBL)

An extension of the BWB model that uses VPD instead of relative humidity.

**Note:** The BBL model is currently under development and will be available in a future release.

## Fitting Stomatal Conductance Models

### Example: Fitting the Medlyn Model

```python
from phytorch.fitting import fit_model
from phytorch.models import Medlyn

# Your measured data
data = {
    'gs': torch.tensor([0.15, 0.25, 0.32, 0.40, 0.35, 0.28]),
    'A': torch.tensor([10, 18, 24, 28, 26, 20]),
    'VPD': torch.tensor([1.2, 1.5, 1.8, 2.0, 2.2, 2.5]),
    'Ca': 400  # Constant atmospheric CO2
}

# Fit the model
result = fit_model(
    model=Medlyn(),
    data=data,
    params_to_fit=['g0', 'g1']
)

print(f"g0 = {result.params['g0']:.4f} mol/m²/s")
print(f"g1 = {result.params['g1']:.2f} √kPa")
```

## Coupled Photosynthesis-Stomatal Conductance

PhyTorch allows coupling photosynthesis and stomatal conductance models:

```python
from phytorch.models import FvCB, Medlyn
from phytorch.coupled import CoupledModel

# Create coupled model
coupled = CoupledModel(
    photosynthesis=FvCB(Vcmax=100, Jmax=180),
    stomatal=Medlyn(g0=0.01, g1=4.0)
)

# Solve for coupled A and gs
result = coupled.solve(
    Ca=400,       # Atmospheric CO2
    temperature=25,
    ppfd=1500,
    VPD=1.5,
    gbw=0.3       # Boundary layer conductance (mol/m²/s)
)

print(f"A = {result['A']:.2f} μmol/m²/s")
print(f"gs = {result['gs']:.3f} mol/m²/s")
print(f"Ci = {result['Ci']:.1f} μmol/mol")
```

## Model Comparison

Different models may be appropriate for different species or conditions:

```python
from phytorch.models import Medlyn, BallWoodrowBerry
from phytorch.evaluation import compare_models

models = {
    'Medlyn': Medlyn(),
    'BWB': BallWoodrowBerry()
}

comparison = compare_models(
    models=models,
    data=validation_data,
    metrics=['RMSE', 'R2', 'AIC']
)
```

## References

- Medlyn, B. E., et al. (2011). Reconciling the optimal and empirical approaches to modelling stomatal conductance. *Global Change Biology*, 17(6), 2134-2144.
- Ball, J. T., Woodrow, I. E., & Berry, J. A. (1987). A model predicting stomatal conductance and its contribution to the control of photosynthesis under different environmental conditions. *Progress in Photosynthesis Research*, 4, 221-224.
- Buckley, T. N., Mott, K. A., & Farquhar, G. D. (2003). A hydromechanical and biochemical model of stomatal conductance. *Plant, Cell & Environment*, 26(10), 1767-1785.
