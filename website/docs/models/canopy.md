---
sidebar_position: 5
---

# Canopy Architecture Models

PhyTorch implements models for characterizing canopy architecture, including leaf angle distribution which describes how leaves are oriented within a plant canopy.

## Leaf Angle Distribution

Leaf angle distribution (LAD) describes the angular distribution of leaf inclination angles within a canopy. This is a critical parameter for understanding light interception, photosynthesis, and canopy radiative transfer.

### Beta Distribution Model

The leaf angle distribution follows a beta distribution parameterized by shape parameters μ and ν:

$$
f(\theta) = \frac{\sin^{\mu-1}(\theta) \cdot \cos^{\nu-1}(\theta)}{B(\mu, \nu) \cdot 90}
$$

where:
- $\theta$ = Leaf inclination angle from horizontal (0-90°)
- $\mu$ = First shape parameter (controls horizontal tendency)
- $\nu$ = Second shape parameter (controls vertical tendency)
- $B(\mu, \nu)$ = Beta function

### Canonical Distribution Types

Following de Wit (1965), leaf angle distributions are classified into six canonical types based on their fitted μ and ν parameters:

| Type | Description | μ | ν | Example Species |
|------|-------------|---|---|-----------------|
| **Planophile** | Mostly horizontal leaves | 2.770 | 1.172 | Oak, broadleaf crops |
| **Erectophile** | Mostly vertical leaves | 1.172 | 2.770 | Grasses, willow |
| **Plagiophile** | Mostly oblique leaves | 3.326 | 3.326 | Many conifers |
| **Extremophile** | Both horizontal and vertical | 0.433 | 0.433 | Some shrubs |
| **Uniform** | Equal distribution | 1.000 | 1.000 | Theoretical |
| **Spherical** | Spherical distribution | 1.101 | 1.930 | Many tree species |

### Usage

```python
from phytorch import fit
from phytorch.models.canopy import LeafAngleDistribution
import numpy as np

# Leaf angle data (degrees from horizontal)
# Can be binned frequency data or individual measurements
data = {
    'theta': np.array([5, 15, 25, 35, 45, 55, 65, 75, 85]),  # Angle bins
    'frequency': np.array([0.05, 0.12, 0.18, 0.22, 0.20, 0.13, 0.07, 0.02, 0.01])  # Relative frequency
}

# Fit the model
model = LeafAngleDistribution()
result = fit(model, data)

# View fitted parameters
print(f"μ = {result.parameters['mu']:.3f}")
print(f"ν = {result.parameters['nu']:.3f}")
print(f"R² = {result.r_squared:.4f}")

# Classify into canonical type
classification = model.classify(result.parameters)
print(f"\nCanopy type: {classification['type']}")
print(f"Distance from canonical: {classification['distance']:.3f}")
print(f"Canonical μ: {classification['canonical_mu']:.3f}")
print(f"Canonical ν: {classification['canonical_nu']:.3f}")

# Plot the distribution
result.plot()
```

### Example: Planophile Canopy

```python
from phytorch import fit
from phytorch.models.canopy import LeafAngleDistribution
import numpy as np

# Simulate planophile canopy (mostly horizontal leaves)
data = {
    'theta': np.array([10, 20, 30, 40, 50, 60, 70, 80]),
    'frequency': np.array([0.25, 0.30, 0.20, 0.12, 0.07, 0.04, 0.01, 0.01])
}

model = LeafAngleDistribution()
result = fit(model, data)

classification = model.classify(result.parameters)
print(f"Canopy type: {classification['type']}")  # Expected: planophile
```

### Example: Erectophile Canopy

```python
from phytorch import fit
from phytorch.models.canopy import LeafAngleDistribution
import numpy as np

# Simulate erectophile canopy (mostly vertical leaves)
data = {
    'theta': np.array([10, 20, 30, 40, 50, 60, 70, 80]),
    'frequency': np.array([0.01, 0.02, 0.05, 0.10, 0.18, 0.25, 0.28, 0.11])
}

model = LeafAngleDistribution()
result = fit(model, data)

classification = model.classify(result.parameters)
print(f"Canopy type: {classification['type']}")  # Expected: erectophile
```

### Parameters

| Parameter | Description | Typical Range | Units | Default |
|-----------|-------------|---------------|-------|---------|
| `mu` | First shape parameter | 0.4-3.5 | - | 1.5 |
| `nu` | Second shape parameter | 0.4-3.5 | - | 1.5 |

### Data Requirements

Leaf angle distribution data can be provided in two formats:

1. **Binned frequency data** (recommended):
   - `theta`: Center of angle bins (degrees, 0-90)
   - `frequency`: Relative frequency in each bin (normalized to sum to 1)

2. **Individual measurements**:
   - `theta`: Individual leaf angle measurements
   - `frequency`: Can be omitted (assumes equal weight)

### Measuring Leaf Angles

Leaf inclination angles are typically measured using:
- Protractors or inclinometers for direct measurement
- Hemispherical photography analysis
- LiDAR-based canopy scanning
- Manual sampling of representative leaves

Best practices:
- Sample 30-50 leaves per canopy for robust estimates
- Stratify sampling across canopy layers if relevant
- Measure from horizontal (0°) to vertical (90°)
- Record multiple canopies per species/treatment for variability

### Interpretation

**Shape Parameter Relationships**:
- $\mu > \nu$: Tendency toward horizontal leaves (planophile)
- $\mu < \nu$: Tendency toward vertical leaves (erectophile)
- $\mu \approx \nu$: Symmetrical distribution (plagiophile, uniform, or spherical)
- Low $\mu$ and $\nu$ ($< 1$): Bimodal distribution (extremophile)

**Ecological Significance**:
- **Planophile** canopies maximize light interception in low-light environments
- **Erectophile** canopies reduce light saturation and overheating in high-light environments
- **Spherical** distributions are common in mature forest canopies
- LAD affects canopy photosynthesis, water use efficiency, and microclimate

### Classification Confidence

The `distance` value in classification results indicates how close the fitted distribution is to the canonical type:
- Distance < 0.5: Strong match to canonical type
- Distance 0.5-1.0: Moderate match
- Distance > 1.0: Weak match, intermediate between types

### Applications

Leaf angle distribution is used in:
- **Radiative transfer models**: Light penetration and absorption
- **Photosynthesis models**: Canopy-scale carbon assimilation
- **Remote sensing**: Vegetation indices and LAI retrieval
- **Crop modeling**: Yield prediction and optimization
- **Climate models**: Surface energy balance

## Custom Parameter Bounds

Constrain parameters based on expected canopy structure:

```python
from phytorch import fit, FitOptions

# Constrain to planophile-like distributions
options = FitOptions(
    bounds={
        'mu': (2.0, 4.0),  # Higher mu favors horizontal
        'nu': (0.5, 2.0)   # Lower nu
    }
)

result = fit(LeafAngleDistribution(), data, options)
```

## Model Comparison

Compare leaf angle distributions across species or treatments:

```python
# Fit multiple canopies
canopies = {
    'Oak': oak_data,
    'Grass': grass_data,
    'Pine': pine_data
}

model = LeafAngleDistribution()
results = {}

for name, data in canopies.items():
    result = fit(model, data)
    classification = model.classify(result.parameters)
    results[name] = classification
    print(f"{name}: {classification['type']} (μ={classification['mu']:.2f}, ν={classification['nu']:.2f})")
```

## References

- de Wit, C. T. (1965). Photosynthesis of Leaf Canopies. *Agricultural Research Reports No. 663*, Pudoc, Wageningen.
- Campbell, G. S. (1986). Extinction coefficients for radiation in plant canopies calculated using an ellipsoidal inclination angle distribution. *Agricultural and Forest Meteorology*, 36(4), 317-321.
- Wang, W. M., Li, Z. L., & Su, H. B. (2007). Comparison of leaf angle distribution functions: Effects on extinction coefficient and fraction of sunlit foliage. *Agricultural and Forest Meteorology*, 143(1-2), 106-122.
