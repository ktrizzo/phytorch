"""Generic curve-fitting models for PhyTorch."""

from .sigmoidal import Sigmoidal
from .rectangular_hyperbola import RectangularHyperbola
from .nonrectangular_hyperbola import NonrectangularHyperbola
from .linear import Linear
from .arrhenius import Arrhenius
from .peaked_arrhenius import PeakedArrhenius
from .gaussian import Gaussian
from .weibull import Weibull
from .beta import Beta

__all__ = [
    'Sigmoidal',
    'RectangularHyperbola',
    'NonrectangularHyperbola',
    'Linear',
    'Arrhenius',
    'PeakedArrhenius',
    'Gaussian',
    'Weibull',
    'Beta'
]
