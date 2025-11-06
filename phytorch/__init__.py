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
from . import util
from . import utilities
from . import photosynthesis_legacy

# Convenience imports (legacy)
from . import photosynthesis_legacy as fvcb

__all__ = [
    "fit",
    "models",
    "core",
    "util",
    "utilities",
    "photosynthesis_legacy",
    "fvcb",
]
