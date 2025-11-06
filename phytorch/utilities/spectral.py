"""
Spectral data analysis utilities for PhyTorch.

Functions for reading spectral reflectance and transmittance data,
computing band averages (PAR, NIR), and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted mean.

    Args:
        values: Array of values
        weights: Array of weights (e.g., solar irradiance)

    Returns:
        Weighted mean
    """
    return np.sum(values * weights) / np.sum(weights)


def get_default_solar_spectrum() -> pd.DataFrame:
    """Get default solar spectrum (AM1.5 or similar).

    Returns:
        DataFrame with columns: wavelength (nm), irradiance (W/m²/nm)

    Note:
        This is a placeholder. Users should provide their own solar spectrum
        data for accurate weighting. Default assumes uniform weighting.
    """
    # Create uniform weighting across typical spectral range
    wavelength = np.arange(350, 2501, 1)
    irradiance = np.ones_like(wavelength, dtype=float)

    return pd.DataFrame({
        'wavelength': wavelength,
        'irradiance': irradiance
    })


def read_spectral_file(
    filepath: Union[str, Path],
    wavelength_col: Union[int, str] = 0,
    value_col: Union[int, str] = 1,
    skiprows: int = 0,
    delimiter: Optional[str] = None
) -> pd.DataFrame:
    """Read a spectral data file.

    Args:
        filepath: Path to spectral data file
        wavelength_col: Column index or name for wavelength (nm)
        value_col: Column index or name for reflectance/transmittance values
        skiprows: Number of rows to skip at file start
        delimiter: Column delimiter (auto-detected if None)

    Returns:
        DataFrame with columns: wavelength, value
    """
    # Read file
    if delimiter:
        df = pd.read_csv(filepath, delimiter=delimiter, skiprows=skiprows)
    else:
        df = pd.read_csv(filepath, skiprows=skiprows)

    # Extract wavelength and value columns
    if isinstance(wavelength_col, int):
        wavelengths = df.iloc[:, wavelength_col].values
    else:
        wavelengths = df[wavelength_col].values

    if isinstance(value_col, int):
        values = df.iloc[:, value_col].values
    else:
        values = df[value_col].values

    return pd.DataFrame({
        'wavelength': wavelengths,
        'value': values
    })


def compute_band_averages(
    wavelength: np.ndarray,
    values: np.ndarray,
    solar_spectrum: Optional[pd.DataFrame] = None,
    par_range: Tuple[float, float] = (400, 700),
    nir_range: Tuple[float, float] = (700, 2500)
) -> Dict[str, float]:
    """Compute PAR and NIR band averages weighted by solar spectrum.

    Args:
        wavelength: Wavelength values (nm)
        values: Reflectance or transmittance values (fraction or %)
        solar_spectrum: DataFrame with 'wavelength' and 'irradiance' columns.
                       If None, uses uniform weighting.
        par_range: Wavelength range for PAR band (nm)
        nir_range: Wavelength range for NIR band (nm)

    Returns:
        Dictionary with keys: 'par', 'nir' containing band averages
    """
    if solar_spectrum is None:
        solar_spectrum = get_default_solar_spectrum()

    # Interpolate solar spectrum to match wavelength grid
    solar_interp = np.interp(
        wavelength,
        solar_spectrum['wavelength'].values,
        solar_spectrum['irradiance'].values,
        left=0, right=0
    )

    # PAR band
    par_mask = (wavelength >= par_range[0]) & (wavelength <= par_range[1])
    par_values = values[par_mask]
    par_weights = solar_interp[par_mask]
    par_avg = weighted_mean(par_values, par_weights)

    # NIR band
    nir_mask = (wavelength >= nir_range[0]) & (wavelength <= nir_range[1])
    nir_values = values[nir_mask]
    nir_weights = solar_interp[nir_mask]
    nir_avg = weighted_mean(nir_values, nir_weights)

    return {
        'par': par_avg,
        'nir': nir_avg
    }


def analyze_spectral_files(
    filepaths: List[Union[str, Path]],
    reflectance_indices: Optional[List[int]] = None,
    transmittance_indices: Optional[List[int]] = None,
    solar_spectrum: Optional[pd.DataFrame] = None,
    wavelength_col: Union[int, str] = 0,
    value_col: Union[int, str] = 1,
    skiprows: int = 0,
    plot: bool = True,
    xlim: Tuple[float, float] = (350, 2000)
) -> Dict:
    """Analyze multiple spectral files and compute band averages.

    Args:
        filepaths: List of paths to spectral data files
        reflectance_indices: Indices of files that are reflectance measurements
        transmittance_indices: Indices of files that are transmittance measurements
        solar_spectrum: Solar spectrum for weighting (uniform if None)
        wavelength_col: Column index/name for wavelength
        value_col: Column index/name for values
        skiprows: Rows to skip when reading files
        plot: Whether to plot spectra
        xlim: X-axis limits for plot (nm)

    Returns:
        Dictionary with keys:
            - 'reflectance_par': List of PAR band reflectance values
            - 'reflectance_nir': List of NIR band reflectance values
            - 'transmittance_par': List of PAR band transmittance values
            - 'transmittance_nir': List of NIR band transmittance values
            - 'reflectance_spectra': List of reflectance DataFrames
            - 'transmittance_spectra': List of transmittance DataFrames
    """
    if reflectance_indices is None:
        reflectance_indices = []
    if transmittance_indices is None:
        transmittance_indices = []

    results = {
        'reflectance_par': [],
        'reflectance_nir': [],
        'transmittance_par': [],
        'transmittance_nir': [],
        'reflectance_spectra': [],
        'transmittance_spectra': []
    }

    # Read all files
    all_spectra = []
    for filepath in filepaths:
        df = read_spectral_file(
            filepath,
            wavelength_col=wavelength_col,
            value_col=value_col,
            skiprows=skiprows
        )
        all_spectra.append(df)

    # Process reflectance files
    for idx in reflectance_indices:
        if idx >= len(all_spectra):
            continue
        df = all_spectra[idx]
        band_avg = compute_band_averages(
            df['wavelength'].values,
            df['value'].values,
            solar_spectrum
        )
        results['reflectance_par'].append(band_avg['par'])
        results['reflectance_nir'].append(band_avg['nir'])
        results['reflectance_spectra'].append(df)

    # Process transmittance files
    for idx in transmittance_indices:
        if idx >= len(all_spectra):
            continue
        df = all_spectra[idx]
        band_avg = compute_band_averages(
            df['wavelength'].values,
            df['value'].values,
            solar_spectrum
        )
        results['transmittance_par'].append(band_avg['par'])
        results['transmittance_nir'].append(band_avg['nir'])
        results['transmittance_spectra'].append(df)

    # Plot if requested
    if plot:
        plot_spectra(
            results['reflectance_spectra'],
            results['transmittance_spectra'],
            xlim=xlim
        )

    # Compute summary statistics
    results['reflectance_par_mean'] = np.mean(results['reflectance_par']) if results['reflectance_par'] else None
    results['reflectance_nir_mean'] = np.mean(results['reflectance_nir']) if results['reflectance_nir'] else None
    results['transmittance_par_mean'] = np.mean(results['transmittance_par']) if results['transmittance_par'] else None
    results['transmittance_nir_mean'] = np.mean(results['transmittance_nir']) if results['transmittance_nir'] else None

    return results


def plot_spectra(
    reflectance_spectra: List[pd.DataFrame],
    transmittance_spectra: List[pd.DataFrame],
    xlim: Tuple[float, float] = (350, 2000),
    ylim: Tuple[float, float] = (0, 60)
) -> plt.Figure:
    """Plot reflectance and transmittance spectra.

    Args:
        reflectance_spectra: List of DataFrames with reflectance data
        transmittance_spectra: List of DataFrames with transmittance data
        xlim: X-axis limits (nm)
        ylim: Y-axis limits (%)

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot reflectance
    ax = axes[0]
    for df in reflectance_spectra:
        ax.plot(df['wavelength'], df['value'], alpha=0.7)
    ax.set_ylabel('Reflectance (%)', fontsize=12)
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_title('Reflectance Spectra', fontsize=13, fontweight='bold')

    # Plot transmittance
    ax = axes[1]
    for df in transmittance_spectra:
        ax.plot(df['wavelength'], df['value'], alpha=0.7)
    ax.set_ylabel('Transmittance (%)', fontsize=12)
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_title('Transmittance Spectra', fontsize=13, fontweight='bold')

    plt.tight_layout()
    return fig


def analyze_single_spectrum(
    filepath: Union[str, Path],
    wavelength_col: Union[int, str] = 0,
    reflectance_col: Union[int, str] = 1,
    transmittance_col: Union[int, str] = 2,
    solar_spectrum: Optional[pd.DataFrame] = None,
    skiprows: int = 0,
    plot: bool = True
) -> Dict:
    """Analyze a single file containing both reflectance and transmittance.

    Args:
        filepath: Path to spectral data file
        wavelength_col: Column index/name for wavelength
        reflectance_col: Column index/name for reflectance
        transmittance_col: Column index/name for transmittance
        solar_spectrum: Solar spectrum for weighting
        skiprows: Rows to skip when reading
        plot: Whether to plot spectra

    Returns:
        Dictionary with PAR and NIR band averages for both reflectance and transmittance
    """
    # Read file
    df = pd.read_csv(filepath, skiprows=skiprows)

    # Extract columns
    if isinstance(wavelength_col, int):
        wavelength = df.iloc[:, wavelength_col].values
    else:
        wavelength = df[wavelength_col].values

    if isinstance(reflectance_col, int):
        reflectance = df.iloc[:, reflectance_col].values
    else:
        reflectance = df[reflectance_col].values

    if isinstance(transmittance_col, int):
        transmittance = df.iloc[:, transmittance_col].values
    else:
        transmittance = df[transmittance_col].values

    # Compute band averages
    refl_bands = compute_band_averages(wavelength, reflectance, solar_spectrum)
    trans_bands = compute_band_averages(wavelength, transmittance, solar_spectrum)

    results = {
        'reflectance_par': refl_bands['par'],
        'reflectance_nir': refl_bands['nir'],
        'transmittance_par': trans_bands['par'],
        'transmittance_nir': trans_bands['nir'],
        'wavelength': wavelength,
        'reflectance': reflectance,
        'transmittance': transmittance
    }

    # Plot if requested
    if plot:
        refl_df = pd.DataFrame({'wavelength': wavelength, 'value': reflectance})
        trans_df = pd.DataFrame({'wavelength': wavelength, 'value': transmittance})
        plot_spectra([refl_df], [trans_df])

    return results
