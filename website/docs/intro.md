---
sidebar_position: 1
---

# Introduction to PhyTorch

**PhyTorch** is a comprehensive physiological plant modeling toolkit built on PyTorch. It provides a unified framework for modeling photosynthesis, stomatal conductance, plant hydraulics, radiative properties, and environmental responses.

## Overview

PhyTorch extends the capabilities of [PhoTorch](https://github.com/GEMINI-Breeding/photorch), a robust photosynthesis model fitting package, to provide a complete suite of plant physiological models. By leveraging PyTorch's automatic differentiation and GPU acceleration, PhyTorch enables efficient parameter estimation and model fitting for complex plant processes.

## Key Features

- **Modular Design**: Flexible, composable model components that can be combined to simulate complex plant processes
- **GPU Acceleration**: Leverage PyTorch for fast computation and automatic differentiation
- **Multiple Models**:
  - Photosynthesis models (FvCB)
  - Stomatal conductance models (BMF, MED, BWB, BBL)
  - Plant hydraulics models
  - Radiative properties (PROSPECT)
  - Environmental response models
- **Easy Parameter Fitting**: Robust optimization routines for estimating model parameters from experimental data
- **Flexible Configuration**: Customize temperature responses, light responses, and other model behaviors

## Why PhyTorch?

PhyTorch is designed for researchers in plant physiology, ecology, and agriculture who need:

1. **Accurate Models**: Implementations based on well-established physiological theories
2. **Fast Computation**: GPU acceleration for fitting models to large datasets
3. **Flexibility**: Easy customization and extension of model components
4. **Integration**: Seamless integration with the PyTorch ecosystem for machine learning applications

## Getting Started

```bash
pip install phytorch
```

Check out the [Installation Guide](./installation.md) for detailed instructions and the [Getting Started Tutorial](./getting-started.md) to fit your first model.
