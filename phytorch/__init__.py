"""
PhyTorch: A Physiological Plant Modeling Toolkit

A PyTorch-based package for modeling plant physiological processes including
photosynthesis, stomatal conductance, leaf hydraulics, and leaf optical properties.
"""

__version__ = "0.1.0"

from . import photosynthesis
from . import stomatalconductance
from . import leafoptics
from . import leafhydraulics
from . import util

# Convenience imports
from . import photosynthesis as fvcb
from . import stomatalconductance as stomatal
from . import leafoptics as prospect

__all__ = [
    "photosynthesis",
    "stomatalconductance",
    "leafoptics",
    "leafhydraulics",
    "util",
    "fvcb",
    "stomatal",
    "prospect",
]
