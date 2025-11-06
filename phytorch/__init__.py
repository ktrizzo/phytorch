"""
PhyTorch: A Physiological Plant Modeling Toolkit

A PyTorch-based package for modeling plant physiological processes including
photosynthesis, stomatal conductance, leaf hydraulics, and leaf optical properties.
"""

__version__ = "0.1.1"

# Main fitting interface
from .fit import fit

# Submodules
from . import models
from . import core
from . import utilities

__all__ = [
    "fit",
    "models",
    "core",
    "utilities",
]
