---
sidebar_position: 2
---

# Community Notebook: Stomatal Conductance Analysis with PhyTorch

This notebook demonstrates fitting stomatal conductance models to understand plant water use efficiency and stomatal behavior.

## Overview

**Goal**: Compare different stomatal conductance models (Medlyn, Ball-Woodrow-Berry, Buckley-Mott-Farquhar) and their ability to predict stomatal responses to environmental conditions

**Dataset**: LI-COR 6800 measurements across a range of VPD and light conditions

## Load and Prepare Data

```python
from phytorch import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load stomatal conductance data
df = pd.read_csv('data/stomatal_measurements.csv')

# Initialize LI-COR data object
lcd = stomatal.initLicordata(df, preprocess=True)

print(f"Loaded {len(df)} measurements")
print(f"VPD range: {lcd.VPD.min():.2f} - {lcd.VPD.max():.2f} kPa")
print(f"gs range: {lcd.Cond.min():.3f} - {lcd.Cond.max():.3f} mol/m²/s")
```

## Fit Multiple Stomatal Conductance Models

```python
# Define models to compare
model_types = ['MED', 'BWB', 'BMF']
model_names = {
    'MED': 'Medlyn (USO)',
    'BWB': 'Ball-Woodrow-Berry',
    'BMF': 'Buckley-Mott-Farquhar'
}

results = {}

# Fit each model
for model_type in model_types:
    print(f"\nFitting {model_names[model_type]} model...")

    # Initialize model
    model = stomatal.model(lcd, model_type=model_type)

    # Fit model
    fitresult = stomatal.fit(model, learn_rate=0.01, maxiteration=10000)

    # Get predictions
    predicted = model.predict(lcd)

    # Calculate R²
    ss_res = np.sum((lcd.Cond - predicted)**2)
    ss_tot = np.sum((lcd.Cond - np.mean(lcd.Cond))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((lcd.Cond - predicted)**2))

    # Store results
    results[model_type] = {
        'model': model,
        'fit': fitresult,
        'predicted': predicted,
        'r2': r_squared,
        'rmse': rmse
    }

    print(f"  R² = {r_squared:.3f}")
    print(f"  RMSE = {rmse:.4f} mol/m²/s")

    # Print fitted parameters
    if model_type == 'MED':
        print(f"  g0 = {fitresult.params['g0']:.4f} mol/m²/s")
        print(f"  g1 = {fitresult.params['g1']:.2f}")
    elif model_type == 'BWB':
        print(f"  g0 = {fitresult.params['g0']:.4f} mol/m²/s")
        print(f"  g1 = {fitresult.params['g1']:.2f}")
```

## Visualize Model Performance

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, model_type in enumerate(model_types):
    ax = axes[idx]

    observed = lcd.Cond
    predicted = results[model_type]['predicted']
    r2 = results[model_type]['r2']
    rmse = results[model_type]['rmse']

    # 1:1 plot
    ax.scatter(observed, predicted, alpha=0.5, s=40)

    # 1:1 line
    min_val = min(observed.min(), predicted.min())
    max_val = max(observed.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1')

    ax.set_xlabel('Observed gs (mol/m²/s)', fontsize=11)
    ax.set_ylabel('Predicted gs (mol/m²/s)', fontsize=11)
    ax.set_title(f"{model_names[model_type]}\nR² = {r2:.3f}, RMSE = {rmse:.4f}",
                 fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Stomatal Response to VPD

```python
# Analyze gs response to VPD using the best model (e.g., Medlyn)
best_model = 'MED'
predicted = results[best_model]['predicted']

# Bin data by VPD
vpd_bins = np.linspace(lcd.VPD.min(), lcd.VPD.max(), 10)
vpd_centers = (vpd_bins[:-1] + vpd_bins[1:]) / 2

gs_mean = []
gs_std = []

for i in range(len(vpd_bins)-1):
    mask = (lcd.VPD >= vpd_bins[i]) & (lcd.VPD < vpd_bins[i+1])
    gs_mean.append(lcd.Cond[mask].mean())
    gs_std.append(lcd.Cond[mask].std())

# Plot gs vs VPD
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot of all data
ax.scatter(lcd.VPD, lcd.Cond, alpha=0.3, s=30, label='Observed', color='blue')

# Binned means with error bars
ax.errorbar(vpd_centers, gs_mean, yerr=gs_std, fmt='o-', linewidth=2,
            markersize=8, capsize=5, label='Binned mean ± SD', color='red')

# Model prediction across VPD range
vpd_range = np.linspace(lcd.VPD.min(), lcd.VPD.max(), 100)
# Note: This would require generating predictions at different VPD values
# with constant A and other conditions

ax.set_xlabel('VPD (kPa)', fontsize=12)
ax.set_ylabel('gs (mol/m²/s)', fontsize=12)
ax.set_title('Stomatal Conductance Response to VPD', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gs_vpd_response.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Calculate Intrinsic Water Use Efficiency (iWUE)

```python
# iWUE = A/gs
# Higher iWUE indicates more carbon gained per unit water lost

lcd['iWUE'] = lcd.Photo / lcd.Cond  # μmol CO2 / mol H2O

# Plot iWUE vs VPD
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(lcd.VPD, lcd.iWUE, alpha=0.5, s=50)

# Fit linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(lcd.VPD, lcd.iWUE)

# Plot regression line
vpd_range = np.linspace(lcd.VPD.min(), lcd.VPD.max(), 100)
ax.plot(vpd_range, slope * vpd_range + intercept, 'r-', linewidth=2,
        label=f'Linear fit: R² = {r_value**2:.3f}, p = {p_value:.4f}')

ax.set_xlabel('VPD (kPa)', fontsize=12)
ax.set_ylabel('iWUE (μmol CO₂ / mol H₂O)', fontsize=12)
ax.set_title('Intrinsic Water Use Efficiency vs VPD', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iwue_vpd.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\niWUE Statistics:")
print(f"  Mean iWUE: {lcd.iWUE.mean():.2f} ± {lcd.iWUE.std():.2f}")
print(f"  Slope with VPD: {slope:.2f} (p = {p_value:.4f})")
```

## Key Insights

### Model Selection

1. **Medlyn Model (MED)**:
   - Uses VPD directly in formulation
   - Generally performs well across species
   - The g1 parameter reflects stomatal sensitivity to VPD
   - Typical g1 values: 2-8 for C3 plants

2. **Ball-Woodrow-Berry (BWB)**:
   - Uses relative humidity
   - Historically important model
   - May perform less well at extreme VPD

3. **Buckley-Mott-Farquhar (BMF)**:
   - More mechanistic, optimization-based
   - Can provide insights into stomatal optimization strategy

### Biological Interpretation

```python
# Extract g1 from Medlyn model
g1 = results['MED']['fit'].params['g1']
g0 = results['MED']['fit'].params['g0']

print(f"\nMedlyn Model Parameters:")
print(f"  g0 (residual conductance): {g0:.4f} mol/m²/s")
print(f"  g1 (stomatal sensitivity): {g1:.2f}")

# Interpret g1
if g1 < 3:
    interpretation = "Water-conservative strategy (low sensitivity to VPD)"
elif g1 > 6:
    interpretation = "Water-spending strategy (high sensitivity to VPD)"
else:
    interpretation = "Intermediate strategy"

print(f"  Interpretation: {interpretation}")
```

### Stomatal Limitations to Photosynthesis

```python
# Calculate stomatal limitation
# l_s = (Ca - Ci) / Ca

lcd['stomatal_limitation'] = (lcd.CO2R - lcd.Ci) / lcd.CO2R

print(f"\nStomatal Limitation:")
print(f"  Mean: {lcd.stomatal_limitation.mean():.3f}")
print(f"  Range: {lcd.stomatal_limitation.min():.3f} - {lcd.stomatal_limitation.max():.3f}")

# Higher values indicate greater stomatal limitation
if lcd.stomatal_limitation.mean() > 0.3:
    print("  ⚠ Substantial stomatal limitation detected")
    print("    Consider if water stress is present")
```

## Model Comparison Table

```python
# Create summary table
summary = pd.DataFrame({
    'Model': [model_names[m] for m in model_types],
    'R²': [results[m]['r2'] for m in model_types],
    'RMSE': [results[m]['rmse'] for m in model_types]
})

print("\nModel Performance Summary:")
print(summary.to_string(index=False))

# Identify best model
best_idx = summary['R²'].idxmax()
print(f"\nBest performing model: {summary.loc[best_idx, 'Model']}")
```

## Recommendations

1. **For drought studies**: Use Medlyn model with g1 as indicator of drought response
2. **For WUE analysis**: Calculate iWUE and examine relationship with environmental factors
3. **For breeding programs**: Select for optimal g1 values balancing carbon gain and water loss
4. **For modeling**: Consider temperature effects on gs (currently not included in basic models)

## Limitations

- Models assume steady-state conditions
- Dynamic stomatal responses (stomatal kinetics) not captured
- Hydraulic signals and hormonal controls not explicitly modeled
- Temperature effects on gs parameters may need consideration

## References

- Medlyn, B. E., et al. (2011). Reconciling the optimal and empirical approaches to modelling stomatal conductance. *Global Change Biology*, 17(6), 2134-2144.
- Ball, J. T., Woodrow, I. E., & Berry, J. A. (1987). A model predicting stomatal conductance. *Progress in Photosynthesis Research*, 4, 221-224.
- Buckley, T. N., Mott, K. A., & Farquhar, G. D. (2003). A hydromechanical and biochemical model of stomatal conductance. *Plant, Cell & Environment*, 26(10), 1767-1785.
- Lei, T., Rizzo, K. T., & Bailey, B. N. (2025). PhoTorch: a robust and generalized biochemical photosynthesis model fitting package.
