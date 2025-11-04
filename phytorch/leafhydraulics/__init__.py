"""
Leaf Hydraulics Module

This module contains models for leaf water transport and hydraulic conductance.
Models are available via the unified API:

    from phytorch import fit
    from phytorch.models.hydraulics import Sigmoidal, SJB2018

    # Fit vulnerability curve
    result = fit(Sigmoidal(), data)

    # Fit pressure-volume curve
    result = fit(SJB2018(), data)

For legacy compatibility, models are also accessible from this module.
"""

from phytorch.models.hydraulics import Sigmoidal, SJB2018

__all__ = ['Sigmoidal', 'SJB2018']
