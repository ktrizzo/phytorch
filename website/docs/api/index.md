---
sidebar_position: 1
---

# API Reference

Complete API reference for PhyTorch modules and functions.

## Core Modules

### `phytorch.models`

Photosynthesis and stomatal conductance models.

#### Photosynthesis Models
- [`FvCB`](./models/fvcb.md) - Farquhar-von Caemmerer-Berry photosynthesis model

#### Stomatal Conductance Models
- [`Medlyn`](./models/medlyn.md) - Medlyn (USO) stomatal conductance model
- [`BallWoodrowBerry`](./models/bwb.md) - Ball-Woodrow-Berry model
- [`BuckleyMottFarquhar`](./models/bmf.md) - Buckley-Mott-Farquhar model
- [`BallBerryLeuning`](./models/bbl.md) - Ball-Berry-Leuning model

### `phytorch.fitting`

Model fitting and parameter estimation.

- [`fit_model()`](./fitting/fit_model.md) - Fit model parameters to data
- [`FitResult`](./fitting/fit_result.md) - Container for fitting results

### `phytorch.temperature`

Temperature response functions.

- [`Arrhenius`](./temperature/arrhenius.md) - Arrhenius temperature response
- [`PeakedArrhenius`](./temperature/peaked_arrhenius.md) - Peaked Arrhenius response
- [`Q10`](./temperature/q10.md) - Q10 temperature response

### `phytorch.coupled`

Coupled photosynthesis-conductance models.

- [`CoupledModel`](./coupled/coupled_model.md) - Coupled A-gs model solver

### `phytorch.utils`

Utility functions and helpers.

- [`convert_units()`](./utils/convert_units.md) - Unit conversion utilities
- [`validate_data()`](./utils/validate_data.md) - Data validation

## Quick Reference

### Common Workflows

#### Fit Photosynthesis Model
```python
from phytorch.models import FvCB
from phytorch.fitting import fit_model

result = fit_model(FvCB(), data, params_to_fit=['Vcmax', 'Jmax'])
```

#### Fit Stomatal Conductance Model
```python
from phytorch.models import Medlyn
from phytorch.fitting import fit_model

result = fit_model(Medlyn(), data, params_to_fit=['g0', 'g1'])
```

#### Coupled Model
```python
from phytorch.coupled import CoupledModel
from phytorch.models import FvCB, Medlyn

coupled = CoupledModel(FvCB(), Medlyn())
result = coupled.solve(Ca=400, temperature=25, ppfd=1500, VPD=1.5)
```

## Index

- [Models](./models/index.md)
- [Fitting](./fitting/index.md)
- [Temperature](./temperature/index.md)
- [Utilities](./utils/index.md)
