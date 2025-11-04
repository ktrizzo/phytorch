---
sidebar_position: 4
---

# Hydraulic Models

PhyTorch implements models for plant water relations, including hydraulic conductance (vulnerability curves) and hydraulic capacitance (pressure-volume curves).

## Hydraulic Conductance (Vulnerability Curves)

Vulnerability curves describe the decline in hydraulic conductance as water potential becomes more negative.

### Sigmoidal Model

The sigmoidal model describes vulnerability to cavitation:

$$
K_{rel} = \frac{K_{max}}{1 + \left|\frac{\psi}{\psi_{50}}\right|^s}
$$

where:
- $K_{rel}$ = Relative hydraulic conductance (0-1)
- $K_{max}$ = Maximum conductance (normalized to 1)
- $\psi$ = Water potential (MPa)
- $\psi_{50}$ = Water potential at 50% loss of conductance (MPa)
- $s$ = Slope parameter

#### Usage

```python
from phytorch import fit
from phytorch.models.hydraulics import Sigmoidal
import numpy as np

# Vulnerability curve data
data = {
    'x': np.array([-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5]),  # Water potential (MPa)
    'psi': np.array([0.98, 0.92, 0.78, 0.51, 0.28, 0.12, 0.05])  # Relative conductance
}

result = fit(Sigmoidal(), data)

print(f"P50: {result.parameters['x50']:.2f} MPa")
print(f"Slope: {result.parameters['s']:.2f}")
print(f"R² = {result.r_squared:.4f}")

# Plot vulnerability curve
result.plot()
```

#### Parameters

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| `Kmax` | Maximum conductance | 0.95-1.0 | normalized |
| `psi50` | P50 value | -0.5 to -8.0 | MPa |
| `s` | Slope parameter | 10-100 | - |

#### Interpretation

- **P50 (ψ₅₀)**: More negative values indicate greater drought tolerance
- **Slope**: Steeper slopes indicate rapid loss of conductance over a narrow water potential range

## Hydraulic Capacitance (Pressure-Volume Curves)

Pressure-volume (P-V) curves characterize cell and tissue water relations.

### SJB2018 Model

The Sack-John-Buckley (2018) model describes P-V relationships:

$$
\psi = p + \pi
$$

where:

$$
p = \pi_o \cdot \max\left(0, \frac{w - w_{tlp}}{1 - w_{tlp}}\right)^{\epsilon}
$$

$$
\pi = -\frac{\pi_o}{w}
$$

- $\psi$ = Water potential (MPa)
- $p$ = Turgor pressure (MPa, positive)
- $\pi$ = Osmotic potential (MPa, negative)
- $w$ = Relative water content (0-1)
- $\pi_o$ = Osmotic pressure at full turgor (MPa, positive value)
- $w_{tlp}$ = Relative water content at turgor loss point
- $\epsilon$ = Bulk modulus of elasticity (MPa)

#### Usage

```python
from phytorch import fit
from phytorch.models.hydraulics import SJB2018
import numpy as np

# Pressure-volume curve data
data = {
    'w': np.array([1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]),  # RWC
    'psi': np.array([-0.1, -0.3, -0.6, -1.0, -1.5, -2.1, -2.8])  # Water potential (MPa)
}

result = fit(SJB2018(), data)

print(f"Osmotic pressure at full turgor: {result.parameters['pi_o']:.2f} MPa")
print(f"Turgor loss point: {result.parameters['w_tlp']:.3f}")
print(f"Bulk modulus: {result.parameters['epsilon']:.2f} MPa")
print(f"R² = {result.r_squared:.4f}")

# Plot P-V curve
result.plot()
```

#### Parameters

| Parameter | Description | Typical Range | Units | Default |
|-----------|-------------|---------------|-------|---------|
| `pi_o` | Osmotic pressure at full turgor | 1.0-3.0 | MPa | 2.0 |
| `w_tlp` | RWC at turgor loss point | 0.70-0.90 | fraction | 0.85 |
| `epsilon` | Bulk modulus of elasticity | 5-30 | MPa | 1.0 |

**Note**: `pi_o` is the osmotic **pressure** (positive value). The corresponding osmotic **potential** at full turgor is `-pi_o` (e.g., default 2.0 MPa pressure = -2.0 MPa potential).

#### Key P-V Parameters

From the fitted curve, you can derive important physiological parameters:

**Turgor Loss Point (TLP)**:
- Water potential at zero turgor: $\psi_{tlp} = -\pi_o / w_{tlp}$
- Lower (more negative) values indicate greater drought tolerance

**Osmotic Pressure at Full Turgor** ($\pi_o$):
- Higher values (representing more negative osmotic potential) indicate greater osmotic adjustment capacity
- Osmotic potential at full turgor = $-\pi_o$

**Bulk Modulus of Elasticity**:
- Higher values indicate stiffer cell walls
- Lower values indicate greater cell wall elasticity

#### Interpretation Example

```python
# Calculate derived parameters
pi_o = result.parameters['pi_o']
w_tlp = result.parameters['w_tlp']
epsilon = result.parameters['epsilon']

# Turgor loss point
psi_tlp = -pi_o / w_tlp
print(f"Turgor loss point: {psi_tlp:.2f} MPa")

# Osmotic potential at full turgor (negative value)
pi_full = -pi_o
print(f"Osmotic potential at full turgor: {pi_full:.2f} MPa")

# Cell wall elasticity interpretation
if epsilon < 10:
    print("Elastic cell walls")
elif epsilon > 20:
    print("Rigid cell walls")
else:
    print("Intermediate cell wall elasticity")
```

## Sign Conventions

PhyTorch follows standard plant physiology conventions:

| Variable | Sign | Range | Meaning |
|----------|------|-------|---------|
| Water potential (ψ) | Negative | 0 to -10 MPa | More negative = drier |
| Turgor pressure (p) | Positive | 0 to 3 MPa | Zero at TLP |
| Osmotic potential (π) | Negative | -0.5 to -5 MPa | More negative = higher solute concentration |
| Osmotic pressure (π_o) | Positive | 0.5 to 5 MPa | Equal to -π in magnitude |

## Custom Parameter Bounds

Constrain parameters based on biological ranges:

```python
from phytorch import fit, FitOptions

# P-V curve with constraints
options = FitOptions(
    bounds={
        'pi_o': (0.5, 5.0),    # Osmotic potential
        'w_tlp': (0.6, 0.95),  # TLP water content
        'epsilon': (1, 50)      # Bulk modulus
    }
)

result = fit(SJB2018(), data, options)
```

## Data Requirements

### Vulnerability Curves
- Measure hydraulic conductance at multiple water potentials
- Typically 6-10 points across the range
- Include points near P50 for accurate estimation
- Normalize conductance to maximum value

### Pressure-Volume Curves
- Measure water potential at multiple relative water contents
- Start from full turgor (RWC ≈ 1.0)
- Continue past turgor loss point (RWC < w_tlp)
- Typically 7-12 measurements

## Plotting

Both models support automatic visualization:

```python
# 1:1 plot + model fit curve
result.plot()

# Save plot
result.plot(save='hydraulic_curve.png', show=False)
```

## Model Comparison

Compare different hydraulic parameters across species or treatments:

```python
# Fit multiple datasets
species_data = {
    'Species A': data_A,
    'Species B': data_B,
    'Species C': data_C
}

results = {}
for species, data in species_data.items():
    results[species] = fit(SJB2018(), data)

# Compare P50 values
for species, result in results.items():
    p50 = result.parameters['x50']
    print(f"{species}: P50 = {p50:.2f} MPa")
```

## References

- Sack, L., John, G. P., & Buckley, T. N. (2018). ABA accumulation in dehydrating leaves is associated with decline in cell volume, not turgor pressure. *Plant Physiology*, 176(1), 489-495.
- Tyree, M. T., & Hammel, H. T. (1972). The measurement of the turgor pressure and the water relations of plants by the pressure-bomb technique. *Journal of Experimental Botany*, 23(1), 267-282.
- Pammenter, N. W., & Vander Willigen, C. (1998). A mathematical and statistical analysis of the curves illustrating vulnerability of xylem to cavitation. *Tree Physiology*, 18(8-9), 589-593.
