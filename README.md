# PhyTorch

**A Comprehensive Physiological Plant Modeling Toolkit**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

PhyTorch is a PyTorch-based package for fitting and analyzing plant physiological models. It provides a unified, simple API for model fitting with efficient GPU-accelerated optimization and automatic differentiation.

## Features

- **Unified API**: Simple `fit(model, data)` interface for all models
- **Photosynthesis Models**: FvCB (Farquhar-von Caemmerer-Berry) model with flexible temperature and light response functions
- **Stomatal Conductance**: Multiple empirical and semi-empirical models (Medlyn, Ball-Woodrow-Berry, Buckley-Mott-Farquhar)
- **Leaf Optical Properties**: PROSPECT model for leaf spectral reflectance and transmittance
- **Leaf Hydraulics**: Pressure-volume curves (SJB2018), sigmoidal vulnerability curves
- **Canopy Architecture**: Leaf angle distribution models with automatic classification into canonical types
- **Generic Models**: Arrhenius, peaked Arrhenius, Weibull, beta distribution, Gaussian, sigmoidal, and hyperbolic functions
- **GPU Acceleration**: Leverage PyTorch for fast parameter optimization on CPU or GPU
- **Automatic Differentiation**: Efficient gradient-based optimization for model fitting
- **Custom Plotting**: Model-specific diagnostic plots with publication-ready styling

## Installation

### From PyPI (recommended)

```bash
pip install phytorch-lib
```

### From source

```bash
git clone https://github.com/ktrizzo/phytorch.git
cd phytorch
pip install -e .
```

## Quick Start

PhyTorch uses a unified `fit(model, data)` API for all models:

### Pressure-Volume Curves

```python
from phytorch import fit
from phytorch.models.hydraulics import SJB2018
import pandas as pd

# Load pressure-volume data
df = pd.read_csv('pv_data.csv')
data = {
    'w': df['RWC'].values,      # Relative water content
    'psi': df['Psi_MPa'].values # Water potential (MPa)
}

# Fit model
model = SJB2018()
result = fit(model, data)

# View results
print(f"πₒ = {result.parameters['pi_o']:.3f} MPa")
print(f"w_tlp = {result.parameters['w_tlp']:.3f}")
print(f"R² = {result.r_squared:.4f}")

# Plot with custom diagnostics
result.plot(save='pv_curve.png')
```

### Leaf Angle Distribution

```python
from phytorch import fit
from phytorch.models.canopy import LeafAngleDistribution, bin_leaf_angles
import pandas as pd

# Load and bin leaf angle measurements
df = pd.read_csv('leaf_angles.csv')
data = bin_leaf_angles(df['zenith'].values, angle_type='zenith')

# Fit model
model = LeafAngleDistribution()
result = fit(model, data)

# Classify canopy architecture
classification = model.classify(result.parameters)
print(f"Canopy type: {classification['type']}")
print(f"R² = {result.r_squared:.4f}")

# Plot with classification
result.plot(save='leaf_angles.png')
```

### Generic Models

```python
from phytorch import fit
from phytorch.models.generic import Arrhenius, Weibull, Beta

# Temperature response with Arrhenius
model = Arrhenius()
data = {'x': temperatures, 'y': rates}
result = fit(model, data)

# Weibull distribution
model = Weibull()
result = fit(model, data)

# Beta distribution
model = Beta()
result = fit(model, data)
```

## Package Structure

```
phytorch/
├── fit.py               # Unified fit() function
├── models/              # Model library
│   ├── generic/         # Generic mathematical models
│   ├── hydraulics/      # Leaf hydraulics models
│   └── canopy/          # Canopy architecture models
├── photosynthesis/      # FvCB photosynthesis models (legacy API)
├── stomatalconductance/ # Stomatal conductance models (legacy API)
├── leafoptics/          # PROSPECT leaf optical properties (legacy API)
├── leafhydraulics/      # Leaf hydraulics (legacy API)
├── core/                # Core optimization and fitting infrastructure
└── data/                # Example datasets
```

## Documentation

Full documentation is available at [https://phytorch.org](https://phytorch.org)

## Citation

If you use PhyTorch in your research, please cite:

```
Lei, T., Rizzo, K. T., & Bailey, B. N. (2025). PhoTorch: A robust and generalized
biochemical photosynthesis model fitting package. (In preparation)
```

## Credits

PhyTorch is an extension and reorganization of [PhoTorch](https://github.com/GEMINI-Breeding/photorch), developed by:
- Tong Lei
- Kyle T. Rizzo
- Brian N. Bailey

## License

PhyTorch is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please [open an issue](https://github.com/ktrizzo/phytorch/issues) on GitHub.
