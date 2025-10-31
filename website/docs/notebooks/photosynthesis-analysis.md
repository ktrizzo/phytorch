---
sidebar_position: 1
---

# Community Notebook: Photosynthesis Analysis with PhyTorch

This notebook demonstrates fitting FvCB photosynthesis models to A-Ci response curves and extracting biological insights.

## Overview

**Goal**: Fit the FvCB model to A-Ci curves from different genotypes to compare photosynthetic capacity

**Dataset**: LI-COR 6800 measurements of A-Ci curves at 25°C and saturating light (1500 μmol/m²/s PPFD)

## Load and Prepare Data

```python
from phytorch import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load A-Ci curve data for multiple genotypes
genotypes = ['Genotype_A', 'Genotype_B', 'Genotype_C']
results = {}

for geno in genotypes:
    df = pd.read_csv(f'data/{geno}_aci.csv')
    lcd = fvcb.initLicordata(df, preprocess=True)
    results[geno] = {'data': lcd}
```

## Fit FvCB Models

```python
# Fit models for each genotype
for geno in genotypes:
    lcd = results[geno]['data']

    # Initialize FvCB model with peaked Arrhenius temperature response
    fvcbm = fvcb.model(lcd, LightResp_type=2, TempResp_type=2)

    # Fit model
    fitresult = fvcb.fit(fvcbm, learn_rate=0.08, maxiteration=20000)

    # Store results
    results[geno]['model'] = fvcbm
    results[geno]['fit'] = fitresult
    results[geno]['predicted'] = fvcbm.predict(lcd)

    # Print fitted parameters
    print(f"\n{geno} Fitted Parameters:")
    print(f"  Vcmax25: {fitresult.params['Vcmax25']:.2f} μmol/m²/s")
    print(f"  Jmax25: {fitresult.params['Jmax25']:.2f} μmol/m²/s")
    print(f"  Rd25: {fitresult.params['Rd25']:.2f} μmol/m²/s")
    print(f"  Tp25: {fitresult.params['Tp25']:.2f} μmol/m²/s")
    print(f"  Jmax/Vcmax: {fitresult.params['Jmax25']/fitresult.params['Vcmax25']:.2f}")
```

## Visualize A-Ci Curves

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, geno in enumerate(genotypes):
    lcd = results[geno]['data']
    predicted = results[geno]['predicted']

    ax = axes[idx]
    ax.scatter(lcd.Ci, lcd.Photo, alpha=0.6, s=80, label='Observed')
    ax.plot(lcd.Ci, predicted, 'r-', linewidth=2, label='Fitted FvCB')
    ax.set_xlabel('Ci (μmol/mol)', fontsize=12)
    ax.set_ylabel('A (μmol/m²/s)', fontsize=12)
    ax.set_title(geno, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('aci_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Compare Photosynthetic Capacity

```python
# Extract key parameters for comparison
params_df = pd.DataFrame({
    'Genotype': genotypes,
    'Vcmax25': [results[g]['fit'].params['Vcmax25'] for g in genotypes],
    'Jmax25': [results[g]['fit'].params['Jmax25'] for g in genotypes],
    'Rd25': [results[g]['fit'].params['Rd25'] for g in genotypes],
    'Tp25': [results[g]['fit'].params['Tp25'] for g in genotypes]
})

# Calculate Jmax/Vcmax ratio
params_df['Jmax_Vcmax'] = params_df['Jmax25'] / params_df['Vcmax25']

# Plot parameter comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

params = ['Vcmax25', 'Jmax25', 'Rd25', 'Jmax_Vcmax']
titles = ['Maximum Carboxylation Rate', 'Maximum Electron Transport',
          'Day Respiration', 'Jmax/Vcmax Ratio']
ylabels = ['Vcmax25 (μmol/m²/s)', 'Jmax25 (μmol/m²/s)',
           'Rd25 (μmol/m²/s)', 'Jmax/Vcmax']

for idx, (param, title, ylabel) in enumerate(zip(params, titles, ylabels)):
    ax = axes[idx // 2, idx % 2]
    ax.bar(params_df['Genotype'], params_df[param], color='steelblue', alpha=0.7)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('parameter_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nParameter Summary:")
print(params_df.to_string(index=False))
```

## Key Insights

### Biological Interpretation

1. **Vcmax vs Jmax**: The ratio of Jmax/Vcmax typically ranges from 1.5-2.5 in C3 plants
   - Values < 1.5 suggest Rubisco limitation
   - Values > 2.5 suggest RuBP regeneration capacity exceeds carboxylation capacity

2. **Vcmax Variation**: Differences in Vcmax among genotypes indicate:
   - Variations in Rubisco content and/or activity
   - Potential for selecting high-capacity genotypes

3. **Rd (Dark Respiration)**:
   - Typically 1-2% of Vcmax25
   - Higher Rd can reduce net carbon gain

### Statistical Analysis

```python
from scipy import stats

# Perform ANOVA to test for significant differences
vcmax_values = [results[g]['fit'].params['Vcmax25'] for g in genotypes]
jmax_values = [results[g]['fit'].params['Jmax25'] for g in genotypes]

# One-way ANOVA
f_stat_vcmax, p_value_vcmax = stats.f_oneway(*[
    [results[g]['fit'].params['Vcmax25']] for g in genotypes
])

print(f"\nVcmax ANOVA: F = {f_stat_vcmax:.3f}, p = {p_value_vcmax:.4f}")

if p_value_vcmax < 0.05:
    print("  ✓ Significant differences detected among genotypes")
else:
    print("  × No significant differences among genotypes")
```

## Limitations Considered

The FvCB model assumes:
- Steady-state photosynthesis
- Uniform light distribution in the leaf
- Well-mixed chloroplast
- No stomatal limitation (using Ci, not Ca)

## Recommendations

1. Collect A-Ci curves across multiple temperatures to estimate temperature response parameters
2. Measure A-Q curves (light response) to validate Jmax estimates
3. Conduct measurements on multiple leaves per genotype for statistical power
4. Consider leaf nitrogen content to explain Vcmax variation

## References

- Farquhar, G. D., von Caemmerer, S., & Berry, J. A. (1980). A biochemical model of photosynthetic CO2 assimilation. *Planta*, 149(1), 78-90.
- Sharkey, T. D., et al. (2007). Fitting photosynthetic carbon dioxide response curves. *Plant, Cell & Environment*, 30(9), 1035-1040.
- Lei, T., Rizzo, K. T., & Bailey, B. N. (2025). PhoTorch: a robust and generalized biochemical photosynthesis model fitting package.
