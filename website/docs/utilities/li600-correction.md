---
sidebar_position: 1
---

# LI-600 Porometer Correction

PhyTorch provides a correction function for systematic bias in stomatal conductance measurements from the LI-COR LI-600 porometer/fluorometer.

## Overview

The LI-600 porometer exhibits systematic positive bias in stomatal conductance measurements due to temperature variations in the instrument's flow path. This correction addresses these measurement errors, which are most significant at:

- High stomatal conductance values (>0.3 mol m⁻² s⁻¹)
- Low humidity conditions
- Large leaf-air temperature differences

The correction is based on solving a coupled system of thermodynamic equations accounting for heat transfer in the instrument's flow path (Rizzo & Bailey, 2025).

## Basic Usage

```python
from phytorch.utilities import correct_LI600

# Correct LI-600 measurements
corrected_data = correct_LI600('li600_data.csv', stomatal_sidedness=1)

# View original vs corrected values
print(corrected_data[['gsw', 'gsw_corrected']])
```

## Parameters

### `correct_LI600(filepath, stomatal_sidedness=1.0, thermal_conductance=0.007, save_output=True, output_path=None)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | str or DataFrame | - | Path to LI-600 CSV file or DataFrame with LI-600 data |
| `stomatal_sidedness` | float | 1.0 | Stomatal distribution factor (see below) |
| `thermal_conductance` | float | 0.007 | Thermal conductance C in W/°C |
| `save_output` | bool | True | Whether to save corrected data to CSV |
| `output_path` | str | None | Output file path (auto-generated if None) |

### Stomatal Sidedness

The `stomatal_sidedness` parameter accounts for stomatal distribution on the leaf:

| Value | Description |
|-------|-------------|
| 1.0 | Hypostomatous (stomata only on lower surface) |
| 2.0 | Amphistomatous (stomata on both surfaces equally) |
| 1.0-2.0 | Intermediate cases |

## Required Data Columns

The LI-600 CSV export must contain these columns:

- `gsw` - Stomatal conductance to water vapor (mol m⁻² s⁻¹)
- `Tref` - Reference temperature (°C)
- `Tleaf` - Leaf temperature (°C)
- `rh_r` - Reference relative humidity (%)
- `rh_s` - Sample relative humidity (%)
- `flow` - Flow rate (µmol s⁻¹)
- `P_atm` - Atmospheric pressure (kPa)
- `E_apparent` - Apparent transpiration rate (mmol m⁻² s⁻¹)

## Output

The function returns a DataFrame with original data plus corrected columns:

| Column | Description | Units |
|--------|-------------|-------|
| `gsw_corrected` | Corrected stomatal conductance | mol m⁻² s⁻¹ |
| `Ta_chamb_corrected` | Corrected chamber temperature | °C |
| `T_in_corrected` | Inlet temperature | °C |
| `T_out_corrected` | Outlet temperature | °C |
| `W_chamb_corrected` | Corrected chamber water vapor mole fraction | mol/mol |
| `stomatal_sidedness` | Applied sidedness value | - |

## Examples

### Hypostomatous Leaves

```python
from phytorch.utilities import correct_LI600

# Correct measurements from leaves with stomata on lower surface only
data = correct_LI600('measurements.csv', stomatal_sidedness=1.0)
```

### Amphistomatous Leaves

```python
# Correct measurements from leaves with stomata on both surfaces
data = correct_LI600('measurements.csv', stomatal_sidedness=2.0)
```

### Without Saving Output

```python
# Process data without saving to file
data = correct_LI600('measurements.csv', save_output=False)
```

### Using DataFrame Input

```python
import pandas as pd
from phytorch.utilities import correct_LI600

# Read and correct data from DataFrame
df = pd.read_csv('measurements.csv')
corrected = correct_LI600(df, stomatal_sidedness=1.5, save_output=False)
```

### Custom Thermal Conductance

```python
# Use custom thermal conductance value (if calibrated for your instrument)
data = correct_LI600('measurements.csv', thermal_conductance=0.008)
```

## Visualization

Plot the correction results to visualize the impact:

```python
from phytorch.utilities import correct_LI600, plot_correction

# Apply correction
corrected_data = correct_LI600('measurements.csv')

# Create comparison plots
fig, axes = plot_correction(corrected_data, save_path='correction_plot.png')
```

The plot shows:
- Left panel: Original vs corrected stomatal conductance
- Right panel: Original vs corrected chamber water vapor mole fraction

## Technical Details

### Correction Method

The correction solves a system of three coupled equations:

1. **Fick's Law** for water vapor diffusion:
   $$E = g_{tw}(W_{leaf} - W_{chamb})$$

2. **Mass Balance** for water vapor:
   $$E = \frac{u_{in}}{s} \frac{W_{out} - W_{in}}{1 - W_{out}}$$

3. **Energy Balance** including latent and sensible heat:
   $$E = \frac{1}{s}\left(\frac{Q + u_{in}h_{in}}{h_{out}} - u_{in}\right)$$

where:
- $E$ = Transpiration rate (mol m⁻² s⁻¹)
- $g_{tw}$ = Total conductance (stomatal + boundary layer)
- $W$ = Water vapor mole fraction (mol/mol)
- $u_{in}$ = Inlet flow rate (mol/s)
- $s$ = Leaf area (m²)
- $Q$ = Heat transfer from air to chamber (W)
- $h$ = Moist air enthalpy (J/mol)

### Convergence

The solver uses `scipy.optimize.fsolve` to find the corrected values. If convergence fails for a data point (residual > 10⁻⁶), that point is assigned zero values in corrected columns and a warning is issued.

Failed convergence typically indicates:
- Invalid or extreme input values
- Instrument malfunction during measurement
- Environmental conditions outside calibration range

## Reference

Rizzo, K.T. & Bailey, B.N. (2025). A psychrometric temperature correction for porometer measurements of stomatal conductance. *(In review)*
