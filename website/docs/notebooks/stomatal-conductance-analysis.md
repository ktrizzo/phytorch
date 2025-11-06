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
from phytorch import fit
from phytorch.models.stomatal import MED2011, BWB1987, BMF2003
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load stomatal conductance data
df = pd.read_csv('data/stomatal_measurements.csv')

# Prepare data dictionary
data = {
    'A': df['Photo'].values,
    'VPD': df['VPD'].values,
    'Ca': df['Ca'].values,
    'gs': df['Cond'].values
}

print(f"Loaded {len(df)} measurements")
print(f"VPD range: {data['VPD'].min():.2f} - {data['VPD'].max():.2f} kPa")
print(f"gs range: {data['gs'].min():.3f} - {data['gs'].max():.3f} mol/m²/s")
```

## Fit Multiple Stomatal Conductance Models

```python
# Define models to compare
models = {
    'MED': (MED2011(), 'Medlyn (USO)'),
    'BWB': (BWB1987(), 'Ball-Woodrow-Berry'),
    'BMF': (BMF2003(), 'Buckley-Mott-Farquhar')
}

results = {}

# Fit each model
for model_key, (model, model_name) in models.items():
    print(f"\nFitting {model_name} model...")

    # Fit model using scipy optimizer
    result = fit(model, data)

    # Calculate RMSE
    rmse = np.sqrt(np.mean(result.residuals**2))

    # Store results
    results[model_key] = {
        'model': model,
        'result': result,
        'r2': result.r_squared,
        'rmse': rmse
    }

    print(f"  R² = {result.r_squared:.3f}")
    print(f"  RMSE = {rmse:.4f} mol/m²/s")

    # Print fitted parameters
    if model_key == 'MED':
        print(f"  g0 = {result.parameters['g0']:.4f} mol/m²/s")
        print(f"  g1 = {result.parameters['g1']:.2f}")
    elif model_key == 'BWB':
        print(f"  g0 = {result.parameters['g0']:.4f} mol/m²/s")
        print(f"  g1 = {result.parameters['g1']:.2f}")
```

## Visualize Model Performance

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (model_key, (_, model_name)) in enumerate(models.items()):
    ax = axes[idx]

    observed = data['gs']
    predicted = results[model_key]['result'].predictions
    r2 = results[model_key]['r2']
    rmse = results[model_key]['rmse']

    # 1:1 plot
    ax.scatter(observed, predicted, alpha=0.5, s=40)

    # 1:1 line
    min_val = min(observed.min(), predicted.min())
    max_val = max(observed.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1')

    ax.set_xlabel('Observed gs (mol/m²/s)', fontsize=11)
    ax.set_ylabel('Predicted gs (mol/m²/s)', fontsize=11)
    ax.set_title(f"{model_name}\nR² = {r2:.3f}, RMSE = {rmse:.4f}",
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
predicted = results[best_model]['result'].predictions

# Bin data by VPD
vpd_bins = np.linspace(data['VPD'].min(), data['VPD'].max(), 10)
vpd_centers = (vpd_bins[:-1] + vpd_bins[1:]) / 2

gs_mean = []
gs_std = []

for i in range(len(vpd_bins)-1):
    mask = (data['VPD'] >= vpd_bins[i]) & (data['VPD'] < vpd_bins[i+1])
    gs_mean.append(data['gs'][mask].mean())
    gs_std.append(data['gs'][mask].std())

# Plot gs vs VPD
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot of all data
ax.scatter(data['VPD'], data['gs'], alpha=0.3, s=30, label='Observed', color='blue')

# Binned means with error bars
ax.errorbar(vpd_centers, gs_mean, yerr=gs_std, fmt='o-', linewidth=2,
            markersize=8, capsize=5, label='Binned mean ± SD', color='red')

# Model prediction across VPD range
vpd_range = np.linspace(data['VPD'].min(), data['VPD'].max(), 100)
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

df['iWUE'] = df['Photo'] / df['Cond']  # μmol CO2 / mol H2O

# Plot iWUE vs VPD
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(df['VPD'], df['iWUE'], alpha=0.5, s=50)

# Fit linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(df['VPD'], df['iWUE'])

# Plot regression line
vpd_range = np.linspace(df['VPD'].min(), df['VPD'].max(), 100)
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
print(f"  Mean iWUE: {df['iWUE'].mean():.2f} ± {df['iWUE'].std():.2f}")
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
g1 = results['MED']['result'].parameters['g1']
g0 = results['MED']['result'].parameters['g0']

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

# Note: Ci would need to be in the dataset - assuming it's available
if 'Ci' in df.columns:
    df['stomatal_limitation'] = (df['Ca'] - df['Ci']) / df['Ca']

    print(f"\nStomatal Limitation:")
    print(f"  Mean: {df['stomatal_limitation'].mean():.3f}")
    print(f"  Range: {df['stomatal_limitation'].min():.3f} - {df['stomatal_limitation'].max():.3f}")

    # Higher values indicate greater stomatal limitation
    if df['stomatal_limitation'].mean() > 0.3:
        print("  ⚠ Substantial stomatal limitation detected")
        print("    Consider if water stress is present")
else:
    print("\nNote: Ci data not available in dataset for stomatal limitation calculation")
```

## Model Comparison Table

```python
# Create summary table
summary = pd.DataFrame({
    'Model': [model_name for _, model_name in models.values()],
    'R²': [results[m]['r2'] for m in results.keys()],
    'RMSE': [results[m]['rmse'] for m in results.keys()]
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
