"""
PhyTorch Utilities

Utility functions for data preprocessing, corrections, and analysis.

Available Functions:
    correct_LI600: Apply Rizzo & Bailey (2025) correction to LI-600 porometer data
    plot_correction: Visualize correction results
    solve_for_A: Solve for Ci and A from porometer data and FvCB parameters
    analyze_spectral_files: Analyze multiple spectral data files
    analyze_single_spectrum: Analyze single file with reflectance and transmittance
    compute_band_averages: Compute PAR and NIR band averages
    plot_spectra: Plot reflectance and transmittance spectra
"""

from .correct_LI600 import correct_LI600, plot_correction
from .porometer import solve_for_A
from .spectral import (
    analyze_spectral_files,
    analyze_single_spectrum,
    compute_band_averages,
    plot_spectra,
    read_spectral_file,
    weighted_mean,
    get_default_solar_spectrum
)

__all__ = [
    'correct_LI600',
    'plot_correction',
    'solve_for_A',
    'analyze_spectral_files',
    'analyze_single_spectrum',
    'compute_band_averages',
    'plot_spectra',
    'read_spectral_file',
    'weighted_mean',
    'get_default_solar_spectrum'
]
