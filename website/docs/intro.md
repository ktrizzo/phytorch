---
sidebar_position: 1
---

# Introduction to PhyTorch

**PhyTorch** is a robust parameter estimation toolkit for plant physiology built on PyTorch. Extract physiological parameters from gas exchange and spectral data using state-of-the-art optimization algorithms.

## What is PhyTorch?

PhyTorch is **not a plant simulator** – it's a **model fitting toolkit** designed to extract meaningful physiological parameters from experimental data. Whether you have A-Ci curves from a LI-COR 6800 or leaf reflectance spectra, PhyTorch helps you estimate:

- **Vcmax, Jmax, TPU, Rd** from photosynthesis measurements (FvCB model)
- **g₀, g₁** stomatal sensitivity parameters (Medlyn, Ball-Berry models)
- **N, Cab, Car, LMA** leaf optical properties (PROSPECT model)
- **Temperature response parameters** (activation energies, optimal temperatures)

## Overview

PhyTorch extends the capabilities of [PhoTorch](https://github.com/GEMINI-Breeding/photorch), a robust photosynthesis model fitting package, to provide a complete suite of plant physiological parameter estimation tools. By leveraging PyTorch's automatic differentiation and GPU acceleration, PhyTorch enables efficient and robust fitting of complex physiological models to experimental data.

## Key Features

- **Robust Fitting**: Automatic constraint enforcement and parameter validation prevent unphysical solutions
- **GPU-Accelerated**: Fit hundreds of A-Ci curves simultaneously using GPU parallelization
- **Automatic Differentiation**: Handle complex models (10+ parameters) without manual gradient derivation
- **Multiple Models**:
  - **Photosynthesis**: FvCB model with flexible temperature and light responses
  - **Stomatal Conductance**: Medlyn (USO), Ball-Woodrow-Berry, Buckley-Mott-Farquhar
  - **Leaf Optics**: PROSPECT-D for leaf reflectance and transmittance
  - **Future**: Hydraulic conductance, C4 photosynthesis
- **From Data to Parameters**: Load LI-COR files directly and extract fitted parameters with uncertainty estimates
- **Research-Ready**: Validated against published datasets and used in active breeding programs

## Who Should Use PhyTorch?

PhyTorch is designed for researchers in plant physiology, ecology, and agriculture who need to:

1. **Extract Parameters**: Estimate Vcmax, Jmax, g₁, and other physiological traits from gas exchange data
2. **Handle Large Datasets**: Fit models to hundreds or thousands of A-Ci curves efficiently
3. **Ensure Robustness**: Avoid unphysical parameter estimates through automatic constraint enforcement
4. **Accelerate Research**: Spend less time debugging optimization, more time analyzing results
5. **Enable High-Throughput Phenotyping**: Process large-scale breeding trial data rapidly

## Getting Started

```bash
pip install phytorch
```

Check out the [Installation Guide](./installation.md) for detailed instructions and the [Getting Started Tutorial](./getting-started.md) to fit your first model.
