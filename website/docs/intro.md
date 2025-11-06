---
sidebar_position: 1
---

# Introduction to PhyTorch

**PhyTorch** is a unified Python toolkit for fitting plant physiology models, from simple curve fitting to complex biochemical processes. With one consistent API, fit any model using the same simple pattern: `fit(model, data, options)`.

## What is PhyTorch?

PhyTorch provides a **unified framework** for parameter estimation across all domains of plant physiology. Whether you're fitting a simple linear regression or a complex photosynthesis model, PhyTorch uses the same intuitive interface:

```python
from phytorch import fit

result = fit(model, data)
result.plot()
result.write()
```

The same pattern works for:
- **Generic curve fitting** (9 models: linear, sigmoidal, Michaelis-Menten, temperature responses, distributions)
- **Hydraulic models** (vulnerability curves, pressure-volume relationships)
- **Photosynthesis models** (biochemical C3 photosynthesis with environmental responses)
- **Stomatal conductance models** (empirical and semi-empirical gs models)
- **Canopy architecture models** (leaf angle distribution with canonical type classification)

## Philosophy

PhyTorch is built on three core principles:

1. **Unified API**: One consistent interface for all models eliminates the need to learn different fitting procedures
2. **Simple to Complex**: Start with basic curve fitting, scale up to multi-parameter physiological models seamlessly
3. **Automatic Intelligence**: Built-in parameter initialization, plotting, and validation work automatically for all models

## Overview

Building on [PhoTorch](https://github.com/GEMINI-Breeding/photorch), PhyTorch extends robust photosynthesis modeling to a comprehensive plant physiology toolkit. By leveraging PyTorch's automatic differentiation and optimization algorithms, PhyTorch delivers efficient, reliable parameter estimation across the full spectrum of plant physiological processes.

## Key Features

- **Unified API**: One function (`fit`) works for all models - from linear regression to complex photosynthesis
- **Comprehensive Model Library**:
  - 9 generic curve-fitting models
  - 2 hydraulic models (vulnerability and P-V curves)
  - Photosynthesis (FvCB with temperature and light responses)
  - Stomatal conductance (Medlyn, Ball-Berry, and more)
  - Canopy architecture (leaf angle distribution with classification)
- **Automatic Visualization**: Built-in plotting adapts to model type (1D curves, 3D surfaces for photosynthesis)
- **Smart Initialization**: Auto-generates parameter starting values from your data
- **GPU-Accelerated**: Leverages PyTorch for fast optimization and batch processing
- **Modular Design**: Easily extend with custom models following the same base class pattern

## Who Should Use PhyTorch?

PhyTorch is designed for researchers in plant physiology, ecology, and agriculture who need to:

1. **Fit Physiological Models Simply**: One unified interface for all model types eliminates API complexity
2. **Scale from Simple to Complex**: Start with basic curve fitting, graduate to multi-parameter models without learning new syntax
3. **Process Data Efficiently**: GPU acceleration and batch processing for high-throughput analysis
4. **Visualize Results Automatically**: Built-in adaptive plotting for immediate quality assessment
5. **Ensure Reproducibility**: Consistent API across models promotes standardized analysis workflows

## Getting Started

```bash
pip install phytorch-lib
```

Check out the [Installation Guide](./installation.md) for detailed instructions and the [Getting Started Tutorial](./getting-started.md) to fit your first model.
