---
sidebar_position: 2
---

# Installation

## Requirements

PhyTorch requires Python 3.8 or later and the following dependencies:

- PyTorch
- NumPy
- SciPy
- Pandas
- Matplotlib

## Installation via pip (Recommended)

First, install the required dependencies:

```bash
pip install torch numpy scipy pandas matplotlib
```

Then install PhyTorch:

```bash
pip install phytorch
```

## Installation via Conda

You can also install PhyTorch using conda:

```bash
conda install pytorch numpy scipy pandas matplotlib -c pytorch -c conda-forge
pip install phytorch
```

## Installation from Source

For development or to access the latest features, you can install from source:

```bash
git clone https://github.com/PlantSimulationLab/phytorch.git
cd phytorch
pip install -e .
```

## GPU Support

PhyTorch leverages PyTorch for GPU acceleration. To enable GPU support, ensure you have a CUDA-compatible GPU and install the appropriate PyTorch version:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for more options.

## Verify Installation

You can verify your installation by importing PhyTorch:

```python
from phytorch import *
print("PhyTorch installed successfully!")
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install torch numpy scipy pandas matplotlib
```

### GPU Not Detected

If PyTorch is not detecting your GPU:

```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
```

If this returns `False`, you may need to reinstall PyTorch with CUDA support.

## Next Steps

Once installed, check out the [Getting Started Guide](./getting-started.md) to fit your first model!
