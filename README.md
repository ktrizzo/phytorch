# PhyTorch

**A Comprehensive Physiological Plant Modeling Toolkit**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

PhyTorch is a PyTorch-based package for modeling plant physiological processes. It provides efficient, GPU-accelerated implementations of models for photosynthesis, stomatal conductance, leaf hydraulics, and optical properties.

## Features

- **Photosynthesis Models**: FvCB (Farquhar-von Caemmerer-Berry) model with flexible temperature and light response functions
- **Stomatal Conductance**: Multiple empirical and semi-empirical models (Medlyn, Ball-Woodrow-Berry, Buckley-Mott-Farquhar)
- **Leaf Optical Properties**: PROSPECT model for leaf spectral reflectance and transmittance
- **Leaf Hydraulics**: *Coming soon* - Models for leaf water transport and hydraulic conductance
- **GPU Acceleration**: Leverage PyTorch for fast parameter optimization on CPU or GPU
- **Automatic Differentiation**: Efficient gradient-based optimization for model fitting

## Installation

### From PyPI (recommended)

```bash
pip install phytorch
```

### From source

```bash
git clone https://github.com/PlantSimulationLab/phytorch.git
cd phytorch
pip install -e .
```

## Quick Start

### Photosynthesis (FvCB Model)

```python
from phytorch import photosynthesis as fvcb
import pandas as pd

# Load your A-Ci curve data
df = pd.read_csv('your_aci_data.csv')
lcd = fvcb.initLicordata(df, preprocess=True)

# Initialize and fit the FvCB model
model = fvcb.model(lcd, LightResp_type=2, TempResp_type=2)
result = fvcb.fit(model, learn_rate=0.08, maxiteration=20000)

# View fitted parameters
print(f"Vcmax25 = {result.params['Vcmax25']:.2f} μmol/m²/s")
print(f"Jmax25 = {result.params['Jmax25']:.2f} μmol/m²/s")
```

### Stomatal Conductance

```python
from phytorch import stomatalconductance as stomatal
import pandas as pd

# Load stomatal conductance data
df = pd.read_csv('your_gs_data.csv')
scd = stomatal.initscdata(df, preprocess=True)

# Fit Medlyn model
model = stomatal.model(scd, model_type='MED')
result = stomatal.fit(model, learn_rate=0.01, maxiteration=10000)

# View fitted parameters
print(f"g0 = {result.params['g0']:.4f} mol/m²/s")
print(f"g1 = {result.params['g1']:.2f}")
```

## Package Structure

```
phytorch/
├── photosynthesis/      # FvCB photosynthesis models
├── stomatalconductance/ # Stomatal conductance models
├── leafoptics/          # PROSPECT leaf optical properties
├── leafhydraulics/      # Leaf hydraulics (under development)
├── data/                # Example datasets
└── util.py              # Utility functions
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

For questions or issues, please [open an issue](https://github.com/PlantSimulationLab/phytorch/issues) on GitHub.
