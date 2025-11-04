---
sidebar_position: 1
---

# Generic Models

PhyTorch provides a comprehensive library of generic curve-fitting models that can be applied across various scientific domains. These models use the same unified API as all PhyTorch models.

## Linear Model

Simple linear regression:

$$
y = a \cdot x + b
$$

### Usage

```python
from phytorch import fit
from phytorch.models.generic import Linear
import numpy as np

data = {
    'x': np.array([1, 2, 3, 4, 5]),
    'y': np.array([2.1, 3.9, 6.2, 7.8, 10.1])
}

result = fit(Linear(), data)
print(f"Slope: {result.parameters['a']:.2f}")
print(f"Intercept: {result.parameters['b']:.2f}")

result.plot()
```

### Parameters

| Parameter | Description | Units |
|-----------|-------------|-------|
| `a` | Slope | - |
| `b` | Intercept | - |

## Sigmoidal Model

Rational sigmoid curve for S-shaped responses:

$$
y = \frac{y_{max}}{1 + \left|\frac{x}{x_{50}}\right|^s}
$$

### Usage

```python
from phytorch import fit
from phytorch.models.generic import Sigmoidal

data = {
    'x': np.array([-3, -2, -1, 0, 1, 2, 3]),
    'y': np.array([0.5, 1.2, 2.5, 5.0, 7.5, 9.0, 9.8])
}

result = fit(Sigmoidal(), data)
result.plot()
```

### Parameters

| Parameter | Description | Units |
|-----------|-------------|-------|
| `ymax` | Maximum response | - |
| `x50` | Half-saturation point | - |
| `s` | Steepness parameter | - |

## Rectangular Hyperbola

Michaelis-Menten kinetics:

$$
y = \frac{V_{max} \cdot x}{K_m + x}
$$

### Usage

```python
from phytorch import fit
from phytorch.models.generic import RectangularHyperbola

# Enzyme kinetics data
data = {
    'x': np.array([0.5, 1, 2, 5, 10, 20, 50]),
    'y': np.array([2.5, 4.5, 7.5, 12, 15, 17, 19])
}

result = fit(RectangularHyperbola(), data)
print(f"Vmax: {result.parameters['Vmax']:.2f}")
print(f"Km: {result.parameters['Km']:.2f}")
```

### Parameters

| Parameter | Description | Units |
|-----------|-------------|-------|
| `Vmax` | Maximum velocity | - |
| `Km` | Michaelis constant | - |

## Non-rectangular Hyperbola

More flexible saturation curve:

$$
y = \frac{\alpha x + y_{max} - \sqrt{(\alpha x + y_{max})^2 - 4\theta\alpha x y_{max}}}{2\theta}
$$

### Usage

```python
from phytorch import fit
from phytorch.models.generic import NonrectangularHyperbola

result = fit(NonrectangularHyperbola(), data)
```

### Parameters

| Parameter | Description | Units |
|-----------|-------------|-------|
| `ymax` | Maximum response | - |
| `alpha` | Initial slope | - |
| `theta` | Curvature (0-1) | - |

## Arrhenius Model

Temperature response following Arrhenius kinetics:

$$
y = y_{ref} \cdot \exp\left[\frac{H_a}{R} \left(\frac{1}{T_{ref}} - \frac{1}{x}\right)\right]
$$

where $R = 0.008314$ kJ/(mol·K) and $T_{ref} = 298.15$ K.

### Usage

```python
from phytorch import fit
from phytorch.models.generic import Arrhenius

# Temperature in Kelvin
data = {
    'x': np.array([283, 288, 293, 298, 303, 308]),
    'y': np.array([50, 70, 95, 125, 160, 200])
}

result = fit(Arrhenius(), data)
print(f"Activation energy: {result.parameters['Ha']:.1f} kJ/mol")
```

### Parameters

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| `yref` | Value at reference T | - | - |
| `Ha` | Activation energy | 30-100 | kJ/mol |

## Peaked Arrhenius Model

Temperature response with high-temperature deactivation:

$$
y = y_{max} \cdot f_{arr}(x) \cdot f_{peak}(x)
$$

### Usage

```python
from phytorch import fit
from phytorch.models.generic import PeakedArrhenius

# Temperature response with optimum
data = {
    'x': np.linspace(280, 320, 20),
    'y': measured_response
}

result = fit(PeakedArrhenius(), data)
print(f"Optimal temperature: {result.parameters['Topt']:.1f} K")
```

### Parameters

| Parameter | Description | Units |
|-----------|-------------|-------|
| `ymax` | Maximum at Topt | - |
| `Ha` | Activation energy | kJ/mol |
| `Hd` | Deactivation energy | kJ/mol |
| `Topt` | Optimal temperature | K |

## Gaussian Model

Bell-shaped curve:

$$
y = a \cdot \exp\left[-\frac{(x - \mu)^2}{2\sigma^2}\right]
$$

### Usage

```python
from phytorch import fit
from phytorch.models.generic import Gaussian

result = fit(Gaussian(), data)
print(f"Peak location: {result.parameters['mu']:.2f}")
print(f"Width: {result.parameters['sigma']:.2f}")
```

### Parameters

| Parameter | Description | Units |
|-----------|-------------|-------|
| `a` | Peak height | - |
| `mu` | Peak location | - |
| `sigma` | Width | - |

## Weibull Distribution

Weibull probability density function:

$$
y = \frac{k}{\lambda}\left(\frac{x - x_0}{\lambda}\right)^{k-1}\exp\left[-\left(\frac{x - x_0}{\lambda}\right)^k\right]
$$

### Usage

```python
from phytorch import fit
from phytorch.models.generic import Weibull

result = fit(Weibull(), data)
```

### Parameters

| Parameter | Description | Units |
|-----------|-------------|-------|
| `x0` | Location parameter | - |
| `lambda` | Scale parameter | - |
| `k` | Shape parameter | - |

## Beta Distribution

Flexible distribution on bounded interval:

$$
y = a \cdot \frac{(x - x_{min})^{\alpha-1}(x_{max} - x)^{\beta-1}}{B(\alpha, \beta)(x_{max} - x_{min})^{\alpha+\beta-1}}
$$

### Usage

```python
from phytorch import fit
from phytorch.models.generic import Beta

result = fit(Beta(), data)
```

### Parameters

| Parameter | Description | Units |
|-----------|-------------|-------|
| `a` | Amplitude | - |
| `alpha` | Shape parameter 1 | - |
| `beta` | Shape parameter 2 | - |
| `xmin` | Lower bound | - |
| `xmax` | Upper bound | - |

## Custom Parameter Bounds

All models support custom parameter constraints:

```python
from phytorch import fit, FitOptions

options = FitOptions(
    bounds={
        'ymax': (0, 100),
        'x50': (-10, 10)
    }
)

result = fit(Sigmoidal(), data, options)
```

## Model Selection

Use information criteria for model comparison:

```python
# Fit multiple models
models = [Linear(), Sigmoidal(), RectangularHyperbola()]
results = [fit(model, data) for model in models]

# Compare R²
for i, result in enumerate(results):
    print(f"Model {i+1}: R² = {result.r_squared:.4f}")
```

## References

- Michaelis, L., & Menten, M. L. (1913). The kinetics of invertase action. *Biochem. z*, 49, 333-369.
- Arrhenius, S. (1889). On the reaction velocity of the inversion of cane sugar by acids. *Z. Phys. Chem*, 4, 226-248.
