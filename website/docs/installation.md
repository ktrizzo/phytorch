---
sidebar_position: 2
---

# Installation

## Requirements

PhyTorch requires Python 3.7 or later. All dependencies (PyTorch, NumPy, SciPy, Pandas, Matplotlib) will be automatically installed.

## Installation via pip (Recommended)

```bash
pip install phytorch-lib
```

## Using with Conda Environments

If you use conda for environment management:

```bash
conda create -n phytorch python=3.9
conda activate phytorch
pip install phytorch-lib
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

### GPU Not Detected

If PyTorch is not detecting your GPU:

```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
```

If this returns `False`, you may need to reinstall PyTorch with CUDA support.

## Next Steps

Once installed, check out the [Getting Started Guide](./getting-started.md) to fit your first model!
